"""On the station table, real size: an X-ACTO KS 1031 manual crank sharpener (121 x 70 x 108 mm), screwed to a pine
2x4 offcut that is clamped to the table, its bore along y at x 30 cm, 118 mm above the table, the dial face toward
the robot's left and the crank on the far face; a new, unsharpened Dixon Ticonderoga My First pencil (10.3 mm,
166 mm, no eraser) lying along y in the two V notches of a pine pen rest to the left.

Goal: sharpen the pencil until the pencil stop ends the cut, then lay it back in its rest and let go. The cut is a
rule (scoring.Cutter): while the pencil's end, through both bore rings, presses the cutting front's feeler in at
least 2 mm and the crank turns forward past its furthest angle within the last turn, the cutting slide moves 1.5 mm
deeper per turn, to 24 mm; while it cuts, the crank meets a cutting drag. See scoring.py for the rule and the tiers.

The sharpener, pencil and rest are mechkit assemblies (mechkit.py next to this file); the cut, the drag and the
crank's whole-turn angle rewrite run in a mechkit on_step callback in the simulator, and the arms are read through
rigstate.py. Neither changes what the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
Variants of the demo (counterfactuals): DEMO_VARIANT=still | backward | let_go | no_layback | table (see demo()).
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built as compound bodies)
import rigstate  # noqa: E402
from scoring import Cutter, Tracker, crank_target, pencil_ends, wrap_shift, wrap_to_pi  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 must hold for 1 s on top of the scorer's own 1 s of release and stillness, so the trial cannot end while a
# gripper holds or touches the pencil.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Sharpen the yellow pencil in the crank sharpener: push it into the hole and turn the crank forward "
               "(its knob going over the top, away from the robot) until the pencil goes in no further. Then pull "
               "the pencil out, lay it back in both notches of its wooden rest and let go.")


# -- shapes ------------------------------------------------------------------------------------------------------
def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ring(y0: float, y1: float, color) -> list:
    """An octagonal ring round the bore axis, 12.3 mm across flats inside, 3 mm thick radially, from y0 to y1:
    eight boxes, the k-th turned k x 45 deg about the axis (local z radial, local x round the ring)."""
    out = []
    r = L.BORE_APOTHEM + L.RING_T / 2
    for k in range(8):
        a = k * math.pi / 4
        out.append(mk.box((L.AXIS_X + r * math.sin(a), (y0 + y1) / 2, L.AXIS_Z + r * math.cos(a)),
                          (L.RING_FACE_W, y1 - y0, L.RING_T), color, quat=mk.matrix_to_quat(_ry(a))))
    return out


def _lead_in(y_face: float, depth: float, color, thick: float) -> list:
    """A 45 deg funnel in front of a ring: eight plates whose inner faces run from BORE_APOTHEM + depth off the axis
    at ``y_face`` to BORE_APOTHEM at ``y_face - depth``."""
    out = []
    slant = depth * math.sqrt(2.0)
    r_mid = L.BORE_APOTHEM + depth / 2 + thick / 2 / math.sqrt(2.0)
    y_mid = y_face - depth / 2 - thick / 2 / math.sqrt(2.0)
    width = 2 * (r_mid + thick / 2) * math.tan(math.pi / 8) + 0.0002
    for k in range(8):
        a = k * math.pi / 4
        rot = _ry(a) @ _rx(math.pi / 4)  # local y along the slant (in and toward the axis), local z its normal
        out.append(mk.box((L.AXIS_X + r_mid * math.sin(a), y_mid, L.AXIS_Z + r_mid * math.cos(a)),
                          (width, slant, thick), color, quat=mk.matrix_to_quat(rot)))
    return out


def _body() -> list:
    """The KS's outer envelope, solid round a 19 x 19 mm passage along the bore axis, on the pine offcut."""
    x0, x1 = L.BODY_X
    y0, y1 = L.BODY_Y
    z0, z1 = L.BODY_Z
    t = L.TUNNEL
    zl, zh = L.AXIS_Z - t, L.AXIS_Z + t
    ylen, yc = y1 - y0, (y0 + y1) / 2
    wall = x1 - t
    return [
        mk.box(*L.BLOCK, L.PINE),
        mk.box((0.0, yc, (z0 + zl) / 2), (x1 - x0, ylen, zl - z0), L.BLACK),  # below the passage
        mk.box((0.0, yc, (zh + z1) / 2), (x1 - x0, ylen, z1 - zh), L.BLACK),  # above it
        mk.box((-(t + wall / 2), yc, L.AXIS_Z), (wall, ylen, 2 * t), L.BLACK),  # its two sides
        mk.box((t + wall / 2, yc, L.AXIS_Z), (wall, ylen, 2 * t), L.BLACK),
        mk.box((0.0, (y0 + L.TUNNEL_Y0) / 2, L.AXIS_Z), (2 * t, L.TUNNEL_Y0 - y0, 2 * t), L.BLACK),  # crank end
        *_ring(*L.DIAL_RING_Y, L.DIAL_GREY),
        *_lead_in(L.DIAL_LEAD_Y[1], L.DIAL_LEAD_Y[1] - L.DIAL_LEAD_Y[0], L.DIAL_GREY, L.LEAD_T),
        *_ring(*L.CARRIAGE_RING_Y, L.CUTTER_GREY),
        *_lead_in(L.CARRIAGE_LEAD_Y[1], L.CARRIAGE_LEAD_Y[1] - L.CARRIAGE_LEAD_Y[0], L.CUTTER_GREY, 0.0015),
    ]


class Sharpener(mk.Assembly):
    """The sharpener exactly as mechkit builds and reads it in the simulator. Only tools/offline/check_task.py calls
    the two methods below: offline it is the offcut and body (without the crank, which the checker cannot turn), and
    its state is rebuilt from the declaration (it is fixed). The offcut's 14.5 mm overhang in y is left out."""

    def offline_box(self):
        lo = np.array([L.BODY_X[0], L.BODY_Y[0], 0.0])  # the body's footprint, down to the table (the offcut
        hi = np.array([L.BODY_X[1], L.BODY_Y[1], L.BODY_Z[1]])  # under it reaches 14.5 mm further each way in y)
        centre = np.asarray(self.pos, dtype=float) + (lo + hi) / 2
        return tuple(float(v) for v in centre), tuple(float(v) for v in hi - lo), float(self.yaw)

    def offline_state(self, state):
        return self.rest_state()


_JOINTS = [
    mk.slider("slide", "-Y", [mk.box(*L.SLIDE_ROD, L.CUTTER_GREY)], upper=L.SLIDE_TRAVEL,
              anchor=(L.AXIS_X, L.SLIDE_ANCHOR_Y, L.AXIS_Z), mass=L.SLIDE_MASS, spring=L.SLIDE_K,
              damping=L.SLIDE_DAMP),
    mk.slider("front", "-Y", [mk.cylinder((L.AXIS_X, L.FRONT_Y0 - L.FRONT_LEN / 2, L.AXIS_Z), L.FRONT_R, L.FRONT_LEN,
                                          "Y", L.CUTTER_GREY)],
              upper=L.FEELER_TRAVEL, anchor=(L.AXIS_X, L.FRONT_Y0 - L.FRONT_LEN / 2, L.AXIS_Z),
              mass=L.FEELER_MASS, spring=L.FEELER_K, damping=L.FEELER_DAMP, parent="slide"),
    mk.hinge("crank", "Y", L.CRANK_ANCHOR,
             [mk.cylinder(L.CRANK_ANCHOR, L.HUB_R, L.HUB_LEN, "Y", L.STEEL), mk.box(*L.CRANK_ARM, L.STEEL)],
             lower=-L.CRANK_LIMIT, upper=L.CRANK_LIMIT, mass=L.CRANK_MASS, spring=L.CRANK_K, damping=L.CRANK_DAMP,
             friction=L.BEARING_FRICTION),
    mk.hinge("knob", "Y", L.KNOB_PIN,
             [mk.cylinder((L.AXIS_X, L.KNOB_CENTRE_Y, L.KNOB_PIN[2]), L.KNOB_R, L.KNOB_LEN, "Y", L.BLACK)],
             lower=-L.KNOB_LIMIT, upper=L.KNOB_LIMIT, mass=L.KNOB_MASS, damping=L.KNOB_DAMP,
             friction=L.KNOB_FRICTION, parent="crank"),
]
SHARPENER = Sharpener(name="sharpener", pos=(L.SH_X, 0.0, L.TOP), color=L.BLACK, mass=None, yaw=0.0,
                      shapes=tuple(_body()), joints=tuple(_JOINTS), friction=L.SHARPENER_FRICTION, **mk.TIGHT_FIT)
SHARPENER._check()

PENCIL = mk.assembly("pencil", origin=(L.PENCIL_START[0], L.PENCIL_START[1], L.TOP + L.PENCIL_START[2]),
                     mass=L.PENCIL_MASS, color=L.YELLOW, friction=L.PENCIL_FRICTION, tight_fit=True,
                     shapes=[mk.cylinder((0.0, 0.0, 0.0), L.PENCIL_R, L.PENCIL_LEN, "Y", L.YELLOW)])


def _v_block(y: float) -> list:
    """One V block of the rest (rest frame): a lower box to the notch's apex, two side boxes up to the top, and two
    cheeks turned +-45 deg about y whose faces form the 90 deg notch."""
    w, d, _ = L.V_BLOCK
    base_top = L.REST_BASE[2]
    h = L.V_HALF_W  # 7 mm: the notch's depth and half its top width
    side_w = w / 2 - h
    face = h * math.sqrt(2.0)  # 9.9 mm along each face
    t = h  # the cheeks' thickness behind their faces
    out = [mk.box((0.0, y, (base_top + L.V_APEX_Z) / 2), (w, d, L.V_APEX_Z - base_top), L.PINE)]
    for sign in (-1.0, 1.0):
        out.append(mk.box((sign * (h + side_w / 2), y, (L.V_APEX_Z + L.V_TOP_Z) / 2), (side_w, d, h), L.PINE))
        mid_x, mid_z = sign * h / 2, (L.V_APEX_Z + L.V_TOP_Z) / 2
        n = np.array([-sign, 1.0]) / math.sqrt(2.0)  # the face's normal, into the notch (x, z)
        cx, cz = mid_x - n[0] * t / 2, mid_z - n[1] * t / 2
        out.append(mk.box((cx, y, cz), (face, d, t), L.PINE, quat=mk.matrix_to_quat(_ry(-sign * math.pi / 4))))
    return out


class Rest(mk.Assembly):
    """The pen rest as mechkit builds it. Offline (tools/offline/check_task.py only) it is a box whose top is at the
    pencil's underside when it lies in both notches, so a pencil the checker drops onto it rests at the seat height."""

    def offline_box(self):
        top = L.SEAT_Z - L.PENCIL_R
        lo, hi = self.local_bounds()
        centre = np.asarray(self.pos, dtype=float) + np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, top / 2])
        return tuple(float(v) for v in centre), (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(top)), 0.0

    def offline_state(self, state):
        return self.rest_state()


_REST_LOCAL_YS = tuple(y - L.REST_XY[1] for y in L.V_BLOCK_YS)
REST = Rest(name="rest", pos=(L.REST_XY[0], L.REST_XY[1], L.TOP), color=L.PINE, mass=None, yaw=0.0,
            shapes=(mk.box((0.0, 0.0, L.REST_BASE[2] / 2), L.REST_BASE, L.PINE),
                    *_v_block(_REST_LOCAL_YS[0]), *_v_block(_REST_LOCAL_YS[1])),
            contact_offset=mk.DEFAULT_CONTACT_OFFSET)
REST._check()

OBJECTS = [SHARPENER, PENCIL, REST]

# -- the rules: the cut, the cutting drag and the crank's whole-turn rewrite ---------------------------------------
CUTTER = Cutter()
_WRAP = {"method": None, "failed": None, "writes": 0}


def _write_joint_positions(values: dict) -> None:
    """Rewrite some of the sharpener's joint positions (rad) in the simulator, velocities unchanged (env 0). Only used
    for whole-turn rewrites of the crank and the knob, which leave every part where it is."""
    import torch

    base = mk.env().unwrapped
    asset = base.scene["sharpener"]
    names = list(asset.joint_names)
    ids = [names.index(j) for j in values]
    q_all = mk._t(asset.data.joint_pos)
    qd_all = mk._t(asset.data.joint_vel)
    pos = torch.tensor([[float(values[j]) for j in values]], dtype=q_all.dtype, device=q_all.device)
    vel = qd_all[:1, ids].clone()
    attempts = (
        ("write_joint_position_to_sim_index", lambda f: f(position=pos, joint_ids=ids)),
        ("write_joint_position_to_sim", lambda f: f(pos, joint_ids=ids)),
        ("write_joint_state_to_sim_index", lambda f: f(position=pos, velocity=vel, joint_ids=ids)),
        ("write_joint_state_to_sim", lambda f: f(pos, vel, joint_ids=ids)),
    )
    if _WRAP["method"] is not None:
        attempts = tuple(a for a in attempts if a[0] == _WRAP["method"])
    errors = []
    for name, call in attempts:
        fn = getattr(asset, name, None)
        if fn is None:
            continue
        try:
            call(fn)
        except Exception as exc:  # noqa: BLE001 - try the next signature, then report all
            errors.append(f"{name}: {exc!r}")
            continue
        if _WRAP["method"] is None:
            print(f"[sharpener] whole-turn rewrite uses {name} (joints {list(values)})", file=sys.stderr, flush=True)
        _WRAP["method"] = name
        _WRAP["writes"] += 1
        return
    raise RuntimeError("no joint-position writer worked: " + ("; ".join(errors) or "none of the methods exist"))


@mk.on_step
def _rules(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics): the cut, then
    the whole-turn rewrite of the crank and the knob, then the slide's and the crank's drive targets."""
    s, p = states.get("sharpener"), states.get("pencil")
    CUTTER.update(s, p)
    try:
        th = float(s.joints["crank"])
        w = float(s.joint_vel.get("crank", 0.0))
        qk = float(s.joints["knob"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    shift, kshift = wrap_shift(th), wrap_shift(qk)
    if (shift or kshift) and _WRAP["failed"] is None:
        values = {}
        if shift:
            values["crank"] = th + shift
        if kshift:
            values["knob"] = qk + kshift
        try:
            _write_joint_positions(values)
        except Exception as exc:  # noqa: BLE001 - loud, once; the crank then stops at its limit (the run shows it)
            _WRAP["failed"] = repr(exc)
            print(f"[sharpener] ERROR: the whole-turn rewrite failed, the crank will stop at +-{L.CRANK_LIMIT} rad: "
                  f"{exc!r}", file=sys.stderr, flush=True)
            shift = 0.0
    elif _WRAP["failed"] is not None:
        shift = 0.0
    mk.set_targets(SHARPENER, "slide", CUTTER.depth)
    mk.set_targets(SHARPENER, "crank", crank_target(th + shift, w, CUTTER.drag))


@mk.on_reset
def _rules_reset() -> None:
    CUTTER.reset()
    OFFLINE.reset()


# -- offline only: the sharpener's joints from the arms (tools/offline/check_task.py has no physics) ---------------
KNOB_HELD_OPENING = 0.3  # the right gripper closed (opening 0..1) ...
KNOB_HELD_DIST = 0.012  # ... with its TCP within 12 mm of the knob's circle (inside, or outside it up to the demo's
KNOB_HELD_TURN = 0.3  # 15 mm pull, below) and within 0.3 rad of the crank's angle, across the knob's axis


class OfflineSharpener:
    """The offline checker carries the pencil with the left gripper and plays the right gripper round the knob's
    circle, but it cannot turn a joint. This rebuilds the sharpener's joints from what it plays: the crank's angle
    from the right TCP while the right gripper is closed on the knob, the slide at the cut depth, and the feeler
    pushed in by the pencil's end (up to its 5 mm travel). Never used in the simulator."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.th = 0.0
        self.prev: float | None = None

    def knob_centre(self) -> np.ndarray:
        return np.array([L.SH_X + L.CRANK_R * math.sin(self.th), L.KNOB_CENTRE_Y,
                         L.TOP + L.AXIS_Z + L.CRANK_R * math.cos(self.th)])

    def state(self, pencil, arms, cutter: Cutter):
        right = arms.get("right") if isinstance(arms, dict) else None
        if right is not None and float(right.opening) <= KNOB_HELD_OPENING:
            tcp = np.asarray(right.tcp, dtype=float)
            dx, dz = tcp[0] - L.SH_X, tcp[2] - (L.TOP + L.AXIS_Z)
            r = math.hypot(dx, dz)
            a = math.atan2(dx, dz)
            on_circle = L.CRANK_R - KNOB_HELD_DIST <= r <= L.CRANK_R + ORBIT_PULL + KNOB_HELD_DIST
            near = abs(wrap_to_pi(a - self.th)) <= KNOB_HELD_TURN
            along = abs(tcp[1] - L.KNOB_CENTRE_Y) <= L.KNOB_LEN / 2
            if on_circle and near and along:
                self.th = a
        w = 0.0 if self.prev is None else wrap_to_pi(self.th - self.prev) / L.DT
        self.prev = self.th
        qf = 0.0
        try:
            end, _ = pencil_ends(pencil)
            ex, ey, ez = end[0] - L.SH_X, end[1], end[2] - L.TOP
            if math.hypot(ex - L.AXIS_X, ez - L.AXIS_Z) < L.ON_AXIS and ey < L.PAST_RINGS_Y:
                qf = min(L.FEELER_TRAVEL, max(0.0, L.FRONT_Y0 - cutter.depth - ey))
        except (AttributeError, TypeError, ValueError):
            qf = 0.0
        origin, quat = SHARPENER.spawn_pose()
        return mk.state_from_poses(SHARPENER, (*origin, *quat), source="offline",
                                   joints={"slide": cutter.depth, "front": qf, "crank": self.th,
                                           "knob": wrap_to_pi(-self.th)},
                                   joint_vel={"crank": w})


OFFLINE = OfflineSharpener()

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The point complete and the pencil back in both notches, released and still for 1 s; held 1 s. This runs
    every control step, so it also feeds the tracker (and, offline, the cut rule)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            CUTTER.reset()
            OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    if mk.env() is None:  # offline (no simulator, so no on_step): the joints from the arms, then the cut rule here
        objects = dict(objects)
        try:
            objects["sharpener"] = OFFLINE.state(objects.get("pencil"), arms, CUTTER)
        except Exception:  # noqa: BLE001 - keep the checker's own state; the cut then sees no press
            pass
        CUTTER.update(objects.get("sharpener"), objects.get("pencil"))
    result = _TRACKER.update(objects, arms, CUTTER)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details, "cut": CUTTER.info,
           "wrap": {"method": _WRAP["method"], "writes": _WRAP["writes"], "failed": _WRAP["failed"]}}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 the point complete (cut depth 24 mm) and the pencil back in both notches, clear of the sharpener,
    released and still for 1 s; 0.75 the point complete; 0.5 cut depth 12 mm; 0.25 cut depth 6 mm; 0 otherwise.
    The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo (open loop: a fixed schedule of TCP poses and finger targets; it reads no simulator state) --
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part (about 20 N a
# finger). Both grips are straight down with the fingers closing along x (yaw 90 deg): round the pencil, which lies
# along y, and round the knob, whose axis is along y.
YAW = -math.pi / 2  # the left grip (joint 4 keeps 19 deg or more from its limit; +90 deg pins joint 6 at the bore)
YAW_R = math.pi / 2  # the right grip (12 deg or more on joint 4 at the top of the knob's circle)
OPEN_PENCIL = -0.015  # 30 mm apart: 10 mm from each side of the 10.3 mm pencil
GRIP_PENCIL = -(2 * L.PENCIL_R - 0.008) / 2  # a 2.3 mm command on the 10.3 mm pencil
OPEN_KNOB = -0.018  # 36 mm apart: 8 mm from each side of the 20 mm knob
GRIP_KNOB = -(2 * L.KNOB_R - 0.008) / 2  # a 12 mm command on the 20 mm knob
GRIP_BACK = 0.020  # the left TCP 20 mm behind the pencil's middle (between the V blocks): 103 mm from its front end
TCP_TO_END = L.HALF + GRIP_BACK
BORE_Z = L.TOP + L.AXIS_Z
# The left arm is compliant (joint stiffness 80 N m/rad shoulder and elbow, 10 wrist: a static calculation for the arm puts
# about 200 N/m at the TCP along y here), so the push on the pencil is set by how far the command leads the pencil:
# the end is commanded LEAD past where it can go, about 4 N on the feeler (4 mm in) with that stiffness, 2.6 mm at
# 120 N/m; a stiffer arm bottoms the feeler, which also counts as pressed.
LEAD = 0.026
KNOB_GRIP_Y = L.KNOB_CENTRE_Y - 0.010  # the right TCP 10 mm outboard of the knob's middle: pads 25 mm off the arm
TURNS = 17  # 16 to the stop, plus one to spare for the crank's lag behind the command
STEPS_PER_TURN = 36  # 1.8 s a turn at full speed: the knob at 0.157 m/s, the crank at 3.49 rad/s
RAMP_STEPS = 20  # linear speed ramps at both ends of the orbit
ORBIT_STEPS = TURNS * STEPS_PER_TURN + RAMP_STEPS  # 632 (the two ramps lose 10 steps of travel each)
# The right arm is compliant too (about 160 N/m at the TCP along x, 400 to 550 along z, from the same static
# calculation), so the crank is carried by how far the command runs ahead of the knob. On the knob's own 45 mm circle
# the weakest point (the knob in front of the axis, going down and back, about 150 deg) gives at most 0.235 N m
# against the 0.2 N m needed, and a 20 % softer arm or a doubled bearing friction stalls the crank there. Commanding a
# circle 15 mm larger (60 mm) raises that to 0.37 N m (0.29 N m with a 20 % softer arm; unit test
# test_compliant_arm_carries_the_crank); the crank's bearing takes the radial part (about 2 N while cutting, at most
# about 6 N with the crank free, which the pads hold by friction). The knob itself stays on its 45 mm circle.
ORBIT_PULL = 0.015
STEP_S = 1.0 / L.CONTROL_HZ
FEED_TURNS = L.TURNS_TO_POINT  # the left arm's command follows the cut for 16 turns, then holds


def orbit_angle(k: int, turns: int = TURNS) -> float:
    """The commanded crank angle (rad) k control steps into the orbit: speed ramps up linearly over RAMP_STEPS,
    holds, and ramps down over the last RAMP_STEPS; the total is ``turns`` whole turns."""
    total = turns * STEPS_PER_TURN + RAMP_STEPS
    w = 2 * math.pi / STEPS_PER_TURN  # rad per step at full speed
    k = min(max(int(k), 0), total)
    r = RAMP_STEPS
    if k <= r:
        return w * k * k / (2 * r)
    if k <= total - r:
        return w * (r / 2 + (k - r))
    m = total - k
    return 2 * math.pi * turns - w * m * m / (2 * r)


def knob_point(phi: float, radius: float = L.CRANK_R) -> tuple[float, float, float]:
    """The right TCP at crank angle ``phi`` (0 = knob up; positive toward +x) on a circle of ``radius`` round the
    crank's axis (the knob's own circle by default)."""
    return (L.SH_X + radius * math.sin(phi), KNOB_GRIP_Y, BORE_Z + radius * math.cos(phi))


def orbit_radius(k: int, turns: int = TURNS) -> float:
    """The commanded circle's radius k control steps into the orbit: the knob's 45 mm plus ORBIT_PULL, blended in
    over the first speed ramp and out over the last, so the orbit starts and ends with the TCP on the knob."""
    total = turns * STEPS_PER_TURN + RAMP_STEPS
    k = min(max(int(k), 0), total)
    blend = min(1.0, k / RAMP_STEPS, (total - k) / RAMP_STEPS)
    return L.CRANK_R + ORBIT_PULL * blend


def orbit_point(k: int, sign: float = 1.0, turns: int = TURNS) -> tuple[float, float, float]:
    """The right TCP's command k control steps into the orbit (``sign`` -1: backward)."""
    return knob_point(sign * orbit_angle(k, turns), orbit_radius(k, turns))


def end_to_tcp_y(end_y: float) -> float:
    """The left TCP's y for the pencil's front end at ``end_y`` (the pencil along y, its front end toward -y)."""
    return end_y + TCP_TO_END


def feed_end_y(turns: float) -> float:
    """The commanded front end's y after ``turns`` commanded crank turns: LEAD past the cutting front's free face,
    following the cut (1.5 mm a turn) for 16 turns."""
    return L.FRONT_Y0 - LEAD - L.CUT_PER_TURN * min(max(turns, 0.0), FEED_TURNS)


def _at(x, y, h):
    return (x, y, L.TOP + h)


PICK = (L.REST_XY[0], L.REST_XY[1] + GRIP_BACK)  # the left TCP over the pencil on its rest
LIFT_H = 0.125  # after letting go: the open finger zone 50 mm above the pencil (30 mm needed)


def _left_before(variant: str) -> list[Waypoint]:
    """Pick the pencil off its rest and press its end onto the cutting front (11.4 s)."""
    px, py = PICK
    plan = [
        Waypoint(1.5, _at(px, py, 0.090), OPEN_PENCIL, "over the pencil, 20 mm behind its middle", YAW),
        Waypoint(1.2, _at(px, py, L.SEAT_Z), OPEN_PENCIL, "down either side of it", YAW),
        Waypoint(0.3, _at(px, py, L.SEAT_Z), OPEN_PENCIL, "settle", YAW),
        Waypoint(1.5, _at(px, py, L.SEAT_Z), GRIP_PENCIL, "close on the pencil", YAW),
        Waypoint(0.3, _at(px, py, L.SEAT_Z), GRIP_PENCIL, "settle the grip", YAW),
        Waypoint(1.2, _at(px, py, L.AXIS_Z), GRIP_PENCIL, "lift it to the bore's height", YAW),
        Waypoint(2.0, (L.SH_X, end_to_tcp_y(0.060), BORE_Z), GRIP_PENCIL,
                 "onto the bore line, its end 19.5 mm off the dial face", YAW),
        Waypoint(0.4, (L.SH_X, end_to_tcp_y(0.060), BORE_Z), GRIP_PENCIL, "settle", YAW),
        Waypoint(2.0, (L.SH_X, end_to_tcp_y(0.014), BORE_Z), GRIP_PENCIL,
                 "slowly into the bore, through the dial ring and into the carriage ring", YAW),
        Waypoint(1.0, (L.SH_X, end_to_tcp_y(feed_end_y(0.0)), BORE_Z), GRIP_PENCIL,
                 "press its end onto the cutting front", YAW),
    ]
    if variant == "let_go":  # counterfactual: let go of the pencil in the bore and stay clear while the crank turns
        y = end_to_tcp_y(feed_end_y(0.0))
        plan += [Waypoint(1.5, (L.SH_X, y, BORE_Z), OPEN_PENCIL, "let go of it in the bore", YAW),
                 Waypoint(1.5, (L.SH_X, y, BORE_Z + 0.080), OPEN_PENCIL, "up and clear", YAW)]
    return plan


def _left_feed(variant: str) -> list[Waypoint]:
    """One waypoint per control step through the orbit: the end's command follows the commanded turns."""
    out = []
    for k in range(1, ORBIT_STEPS + 1):
        n = 0.0 if variant == "still" else orbit_angle(k) / (2 * math.pi)
        out.append(Waypoint(STEP_S, (L.SH_X, end_to_tcp_y(feed_end_y(n)), BORE_Z), GRIP_PENCIL,
                            "keep the pencil pressed on the cut as the crank turns", YAW))
    return out


def _left_after(variant: str) -> list[Waypoint]:
    """Pull the pencil out, carry it back and lay it in both notches, let go and rise clear (10.2 s)."""
    px, py = PICK
    hold = (L.SH_X, end_to_tcp_y(feed_end_y(FEED_TURNS if variant != "still" else 0.0)), BORE_Z)
    plan = [Waypoint(0.5, hold, GRIP_PENCIL, "hold it pressed at the stop", YAW)]
    if variant == "no_layback":
        return plan + [Waypoint(10.0, hold, GRIP_PENCIL, "keep holding it in the sharpener", YAW)]
    place_y = py
    place_x = px if variant != "table" else px - 0.060  # counterfactual: laid on the table 60 mm short of the rest
    place_h = L.SEAT_Z + 0.002 if variant != "table" else L.PENCIL_R + 0.002
    plan += [
        Waypoint(2.0, (L.SH_X, end_to_tcp_y(0.062), BORE_Z), GRIP_PENCIL,
                 "pull it straight out, its end 21.5 mm clear of the dial face", YAW),
        Waypoint(2.2, _at(place_x, place_y, 0.060), GRIP_PENCIL, "carry it over its rest", YAW),
        Waypoint(1.2, _at(place_x, place_y, place_h), GRIP_PENCIL, "lower it to 2 mm above the notches", YAW),
        Waypoint(0.5, _at(place_x, place_y, place_h), GRIP_PENCIL, "settle", YAW),
        Waypoint(1.5, _at(place_x, place_y, place_h), OPEN_PENCIL, "let go", YAW),
        Waypoint(0.3, _at(place_x, place_y, place_h), OPEN_PENCIL, "settle", YAW),
        Waypoint(1.5, _at(place_x, place_y, LIFT_H), OPEN_PENCIL, "straight up, clear of the pencil", YAW),
        Waypoint(0.5, _at(place_x, place_y, LIFT_H), OPEN_PENCIL, "hold", YAW),
    ]
    return plan


def _left(variant: str) -> list[Waypoint]:
    if variant == "let_go":
        return _left_before(variant)
    return _left_before(variant) + _left_feed(variant) + _left_after(variant)


LEFT_PRESS_S = 11.4  # the left arm's pick and press (see _left_before)


def _right(variant: str) -> list[Waypoint]:
    """Grip the knob at the top of its circle, wait for the press, turn the crank 17 times forward (31.6 s), let
    go and rise."""
    top = knob_point(0.0)
    over = (top[0], top[1], BORE_Z + L.CRANK_R + 0.019)  # 19 mm above the grip: joint 4 keeps 7 deg (higher: less)
    plan = [
        Waypoint(1.5, over, OPEN_KNOB, "over the crank knob, pointing up", YAW_R),
        Waypoint(1.2, top, OPEN_KNOB, "down either side of the knob", YAW_R),
        Waypoint(0.3, top, OPEN_KNOB, "settle", YAW_R),
        Waypoint(1.5, top, GRIP_KNOB, "close on the knob", YAW_R),
        Waypoint(0.5, top, GRIP_KNOB, "settle the grip", YAW_R),
    ]
    plan.append(Waypoint(LEFT_PRESS_S - sum(w.duration for w in plan), top, GRIP_KNOB,
                         "wait for the pencil to be pressed", YAW_R))
    sign = -1.0 if variant == "backward" else 1.0
    for k in range(1, ORBIT_STEPS + 1):
        plan.append(Waypoint(STEP_S, orbit_point(k, sign), GRIP_KNOB,
                             "turn the crank forward" if sign > 0 else "turn the crank backward", YAW_R))
    plan += [
        Waypoint(0.5, top, GRIP_KNOB, "hold the knob at the top", YAW_R),
        Waypoint(1.5, top, OPEN_KNOB, "let go of the knob", YAW_R),
        Waypoint(1.5, over, OPEN_KNOB, "up and clear", YAW_R),
    ]
    return plan


def demo():
    """Both arms at once: the left feeds the pencil, the right turns the crank. DEMO_HOLD_ONLY=1: no plan (both arms
    hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactuals): still (the left arm keeps its first press command and does not follow the cut),
    backward (the crank turned backward), let_go (the pencil let go in the bore before the crank turns), no_layback
    (the pencil held in the sharpener to the end), table (laid on the table 60 mm short of the rest)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    variant = os.environ.get("DEMO_VARIANT", "")
    plan = {"left": _left(variant), "right": _right(variant)}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
