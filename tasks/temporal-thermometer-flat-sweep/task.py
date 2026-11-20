"""On the station table: an adult care-training manikin head stands upright on its neck block, facing the robot, on a
sprung neck (pushed straight back at mid-forehead it gives 0.5 N per mm). Its forehead curves round toward each
temple (radius 75 mm in top view) up to the hairline. An Exergen TemporalScanner TAT-5000S (50 x 200 x 30 mm, 213 g;
a 32 mm dome probe head at one end with its sensing cup on the back, the SCAN button on the front under the thumb)
stands on its long edge in a two-block foam rest to the robot's right of the head, cup toward the head.

Goal (the maker's directions): take the scanner, press and hold SCAN, set the cup flush on the centre of the forehead
and slide it along the mid-forehead line round the curve to the hairline at the temple in about 2 to 3 s, keeping it
flush and light, then put the scanner back in its rest, hands off.

The head (a fixed-base mechkit articulation: hinge ``neck``), the scanner (a loose one: slider ``button``) and the
rest are mechkit compounds (mechkit.py, next to this file). The arms are read through rigstate.py (the let-go check).
No software rule moves anything: the springs are the mechkit joints' own drives.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  Counterfactuals (never the submitted demo): DEMO_VARIANT=<name>, see VARIANTS.
Trace:   TASK_TRACE=<file>.
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
from scoring import HEAD, NECK, PROBE  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Take the manikin's temperature with the temporal artery thermometer, then put the thermometer back in its rest."

SKIN = (0.93, 0.78, 0.66)
HAIR = (0.24, 0.16, 0.11)
BROW = (0.35, 0.24, 0.17)
EYE = (0.22, 0.22, 0.26)
LIP = (0.76, 0.45, 0.42)
CHARCOAL = (0.20, 0.21, 0.23)
BODY_COL = (0.90, 0.91, 0.92)
CHROME = (0.74, 0.76, 0.79)
BTN_COL = (0.78, 0.10, 0.10)  # the manual's 'red button'
LED_COL = (0.40, 0.04, 0.04)
FOAM = (0.22, 0.27, 0.36)


# -- the head ---------------------------------------------------------------------------------------------------------
def _tangent_plate(phi: float, r_in: float, thick: float, width: float, z0: float, z1: float, color):
    """A plate lying against the head's curve at angle ``phi`` (0 = the forehead centre, + toward -y), its inner face
    ``r_in`` from the forehead's axis."""
    rc = r_in + thick / 2
    return mk.box((-rc * math.cos(phi), -rc * math.sin(phi), (z0 + z1) / 2), (thick, width, z1 - z0), color,
                  ypr=(phi, 0.0, 0.0))


def head_link_shapes():
    """The nodding head, in the head's frame (origin on the table under the forehead's axis): skull and forehead,
    crown and side hair from the hairline, brows, eyes, nose, lips, jaw, ears and neck."""
    z0, z1 = L.FOREHEAD_Z
    out = [
        mk.cylinder((0.0, 0.0, (z0 + z1) / 2), L.RF, z1 - z0, "Z", SKIN),  # skull: its front is the forehead
        mk.cylinder((0.0, 0.0, (L.Z_HAIR + 0.262) / 2), L.RF + 0.0015, 0.262 - L.Z_HAIR, "Z", HAIR),  # crown hair
        mk.box((0.060, 0.0, 0.180), (0.110, 0.140, 0.140), HAIR),  # back of the head (190 mm front to back)
        mk.cylinder((0.004, 0.0, 0.075), 0.066, 0.080, "Z", SKIN),  # cheeks and jaw
        mk.box((-L.RF - 0.006, 0.0, 0.112), (0.020, 0.020, 0.044), SKIN),  # nose (16 mm proud of the forehead)
        mk.box((-0.064, 0.0, 0.062), (0.008, 0.040, 0.008), LIP),
        mk.cylinder((0.012, 0.0, 0.028), 0.045, 0.036, "Z", SKIN),  # neck
    ]
    w = 2 * (L.RF + 0.003) * math.tan(math.radians(10.0))
    for s in (-1, 1):
        for k in range(6):  # side hair from the hairline (55 degrees) round to the back, 3 mm proud
            phi = s * (L.HAIR_PHI + math.radians(10.0 + 20.0 * k))
            out.append(_tangent_plate(phi, L.RF, 0.003, w, 0.150, L.Z_HAIR + 0.010, HAIR))
        out.append(_tangent_plate(s * math.radians(22.0), L.RF, 0.001, 0.024, L.Z_BROW - 0.004, L.Z_BROW, BROW))
        out.append(_tangent_plate(s * math.radians(25.0), L.RF, 0.001, 0.016, 0.126, 0.134, EYE))
        out.append(mk.box((0.010, s * (L.RF + 0.004), 0.135), (0.030, 0.010, 0.055), SKIN))  # ear
    return out


def _link_com(shapes) -> np.ndarray:
    vols = np.array([s.volume() for s in shapes])
    centres = np.array([s.center for s in shapes], dtype=float)
    return (vols[:, None] * centres).sum(axis=0) / vols.sum()


_HEAD_SHAPES = head_link_shapes()
_COM = _link_com(_HEAD_SHAPES)
NECK_ANCHOR = (float(_COM[0]), 0.0, L.NECK_Z)  # the hinge right under the head's centre of mass: no sag at rest
NECK_SPRING = L.head_spring(float(_COM[2]) - L.NECK_Z)
NECK_JOINT = mk.hinge(NECK, "Y", NECK_ANCHOR, _HEAD_SHAPES, lower=L.HEAD_LOWER, upper=L.HEAD_UPPER,
                      mass=L.HEAD_LINK_MASS, spring=NECK_SPRING, damping=L.HEAD_DAMP)
HEAD_PART = mk.assembly(HEAD, origin=(L.FX, L.FY, L.TOP), color=SKIN, friction=L.FRICTION, tight_fit=True, joints=[NECK_JOINT],
                        shapes=[mk.box((0.012, 0.0, 0.005), (0.130, 0.120, 0.010), CHARCOAL)])  # the neck block's foot

# -- the thermometer ---------------------------------------------------------------------------------------------------
_bx0, _bx1 = L.BODY_X
_by0, _by1 = L.BODY_Y
_ny0, _ny1 = L.NECK_Y
PROBE_SHAPES = [
    # the dome on its angled neck: its cup (face at the origin) faces CUP_DIR, 30 degrees from the back to the head end
    mk.cylinder(tuple(-L.CUP_DEPTH / 2 * c for c in L.CUP_DIR), L.CUP_R, L.CUP_DEPTH, "X", CHROME, ypr=(L.CUP_TILT, 0.0, 0.0)),
    mk.box(((_bx0 + _bx1) / 2, (_ny0 + _ny1) / 2, 0.0), (L.NECK_SIZE_XZ[0], _ny1 - _ny0, L.NECK_SIZE_XZ[1]), CHROME),
    mk.box(((_bx0 + _bx1) / 2, (_by0 + _by1) / 2, 0.0), (_bx1 - _bx0, _by1 - _by0, L.BODY_W), BODY_COL),
    mk.box((_bx0 - 0.00025, (L.DISPLAY_Y[0] + L.DISPLAY_Y[1]) / 2, 0.0), (0.0005, L.DISPLAY_Y[1] - L.DISPLAY_Y[0], 0.024), LED_COL),
]
BUTTON = mk.slider(
    "button", "X", [mk.cylinder((_bx0 - L.BTN_PROUD + 0.003, L.BTN_Y, 0.0), L.BTN_R, 0.006, "X", BTN_COL)],
    upper=L.BTN_TRAVEL, mass=L.BTN_MASS, spring=L.BTN_K, damping=L.BTN_DAMP)
PROBE_PART = mk.assembly(PROBE, origin=(L.PR_X0, L.PR_Y0, L.TOP + L.PR_Z0), mass=L.PR_MASS, joints=[BUTTON],
                         color=BODY_COL, shapes=PROBE_SHAPES, friction=L.FRICTION, tight_fit=True)


def _rest_shapes():
    """Two foam blocks with a 34 mm slot, under the body either side of the grip (thermometer frame, at rest)."""
    out = []
    cx = (_bx0 + _bx1) / 2
    side = (L.REST_OUT_X - L.SLOT_W) / 2
    for y0, y1 in L.REST_BLOCKS:
        cy, ly = (y0 + y1) / 2, y1 - y0
        out.append(mk.box((cx, cy, L.REST_FLOOR / 2), (L.REST_OUT_X, ly, L.REST_FLOOR), FOAM))
        for s in (-1, 1):
            out.append(mk.box((cx + s * (L.SLOT_W / 2 + side / 2), cy, L.REST_TOP / 2), (side, ly, L.REST_TOP), FOAM))
    return out


REST_PART = mk.assembly("rest", origin=(L.HOLD_X, L.HOLD_Y, L.TOP), color=FOAM, friction=L.FRICTION, tight_fit=True,
                        shapes=_rest_shapes())
OBJECTS = [HEAD_PART, REST_PART, PROBE_PART]
SPECS = {PROBE: PROBE_PART, HEAD: HEAD_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# Offline only (never in the simulator). The checker's proxy carries a gripped part rigidly and turns it with the
# gripper; it knows no joints and no contact. This model gives the scorer: (1) the thermometer carried by a gripper
# closed across it at the button (its yaw taken from the checker's carried part), else left where it is (in the rest,
# or on the table); (2) the SCAN button pressed when the pads close on it; (3) the cup stopped by the skin, the head
# nodding back through two springs in series (the arm's sideways give ARM_K, the neck's HEAD_K at the scan line).
GRIP_W = (_bx1 - _bx0) + L.BTN_PROUD  # 29 mm across the button: the pads count as closed on it under this gap


def _gap(opening: float) -> float:
    return float(opening) * rigstate.FULL_GAP


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(state) -> float | None:
    try:
        r = mk.quat_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
        return math.atan2(float(r[1, 0]), float(r[0, 0]))
    except (AttributeError, TypeError, ValueError):
        return None


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.o = np.array([L.PR_X0, L.PR_Y0, L.PR_Z0])  # the cup's centre, above the table
        self.yaw = 0.0
        self.prev_o = self.o.copy()
        self.prev_yaw = 0.0
        self.grip: dict = {}  # side -> the TCP in the thermometer's frame when taken
        self.theta = 0.0
        self.prev_theta = 0.0
        self.button = 0.0

    def head_state(self, theta: float):
        return mk.state_from_poses(HEAD_PART, (L.FX, L.FY, L.TOP, 0.0, 0.0, 0.0, 1.0), joints={NECK: theta},
                                   source="offline")

    def contact(self, cmd: np.ndarray) -> tuple[np.ndarray, float]:
        """Where the cup ends up and the neck's nod for a commanded cup centre ``cmd`` (above the table)."""
        dx, dy = float(cmd[0]) - L.FX, float(cmd[1]) - L.FY
        r = math.hypot(dx, dy)
        on_forehead = L.FOREHEAD_Z[0] + L.CUP_R <= float(cmd[2]) <= L.Z_HAIR - 0.002
        if not on_forehead or r >= L.RF:
            return cmd, 0.0
        phi = math.atan2(-dy, -dx)
        pen = L.RF - r
        c = math.cos(phi)
        force = pen / (1.0 / L.ARM_K + c * c / L.HEAD_K)
        push = min(force * c / L.HEAD_K, L.H_ARM * L.HEAD_UPPER)
        theta = push / L.H_ARM
        drawn = np.array([-L.RF * c, -L.RF * math.sin(phi), float(cmd[2])])  # on the skin, head frame as drawn
        world = self.head_state(theta).link_to_world(NECK, drawn) - np.array([0.0, 0.0, L.TOP])
        return world, theta

    def step(self, arms, proxy_yaw: float | None = None, dt: float = 1.0 / L.CONTROL_HZ) -> None:
        read = rigstate.read_arms(arms)
        self.prev_o, self.prev_yaw, self.prev_theta = self.o.copy(), self.yaw, self.theta
        holder = None
        if read is not None:
            for side, (tcp, _tips, opening) in zip(rigstate.ARMS, read):
                if _gap(opening) < GRIP_W:
                    t = np.asarray(tcp, dtype=float) - np.array([0.0, 0.0, L.TOP])
                    local = rot_z(-self.yaw) @ (t - self.o)
                    near = (abs(local[0] - L.GRIP_LOCAL[0]) <= 0.006 and L.BODY_Y[0] + 0.012 <= local[1] <= L.BODY_Y[1]
                            and abs(local[2]) <= 0.012)
                    if side in self.grip or near:
                        self.grip.setdefault(side, local)
                        holder = side
                        continue
                self.grip.pop(side, None)
        self.button = 0.0
        self.theta = 0.0
        if holder is not None:
            if proxy_yaw is not None:
                self.yaw = proxy_yaw
            t = np.asarray(read[rigstate.ARMS.index(holder)][0], dtype=float) - np.array([0.0, 0.0, L.TOP])
            local = self.grip[holder]
            cmd = t - rot_z(self.yaw) @ local
            self.o, self.theta = self.contact(cmd)
            if abs(local[1] - L.BTN_Y) <= L.BTN_R and abs(local[2]) <= 0.010:
                self.button = L.BTN_TRAVEL
        else:
            d = math.hypot(float(self.o[0]) - L.PR_X0, float(self.o[1]) - L.PR_Y0)
            if d <= 0.012 and abs(math.degrees(self.yaw)) < 15.0 and self.o[2] < L.PR_Z0 + 0.030:
                self.o[2] = L.PR_Z0  # down into the slot
            elif self.o[2] < L.PR_Z0 + 0.030 and d <= 0.030:
                self.o[2] = L.REST_TOP + L.BODY_W / 2  # on the blocks' tops
            else:
                self.o[2] = L.BODY_W / 2  # on the table on its edge

    def states(self, dt: float = 1.0 / L.CONTROL_HZ) -> dict:
        vel = (self.o - self.prev_o) / dt
        w = math.remainder(self.yaw - self.prev_yaw, 2 * math.pi) / dt
        q = mk.yaw_quat(self.yaw)
        pr = mk.state_from_poses(PROBE_PART, (self.o[0], self.o[1], L.TOP + self.o[2], *q),
                                 base_vel=(*vel, 0.0, 0.0, w), joints={"button": self.button}, source="offline")
        hd = mk.state_from_poses(HEAD_PART, (L.FX, L.FY, L.TOP, 0.0, 0.0, 0.0, 1.0), joints={NECK: self.theta},
                                 joint_vel={NECK: (self.theta - self.prev_theta) / dt}, source="offline")
        return {PROBE: pr, HEAD: hd}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(PROBE), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """A clean scan along the forehead's curve from its centre to the hairline, then the thermometer standing back in
    its rest, both hands off it, the head upright, all still for 1 s; held 1 s. This runs first every control step,
    so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    if no_simulator(parts):
        _OFFLINE.step(arms, yaw_of(parts.get(PROBE)))
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
    row["source"] = getattr(parts.get(PROBE), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the head was ever shoved back 8 mm. Else 1.0 a clean scan (flush, on the line, SCAN held, 0.5 to 2.0 N,
    1.5 to 4.0 s for the timed 45 mm) and the thermometer back in its rest, hands off, head upright, still 1 s; 0.75
    the same scan, not put back; 0.5 a scan that was not clean; 0.25 touching for 0.5 s in all; 0 otherwise."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. Only the right arm works;
# the left holds its ready pose. Its fingers close across the scanner's thickness (button and back), so the gripper's
# yaw is the scanner's yaw plus 90 degrees.
OPEN = -0.04695  # 94 mm
OPEN_PR = -(GRIP_W + 0.036) / 2  # 65 mm round the 29 mm at the button
GRIP_PR = -((_bx1 - _bx0) - 0.008) / 2  # 19 mm commanded on the 27 mm body: the pad pushes the button flush
# The cup commanded 5.5 mm into the skin. GPU job 20261120T075550 at 4.5 mm nodded the head only 0.6 to 2.2 mm (the
# touch band is 1 to 4 mm), so contact flickered and the pass was not clean (0.5).
PEN = 0.0065  # 5.5 mm (job 20261120T000005) gave 1.0 to 2.4 mm: right on the 1 mm floor at the centre, so 1 mm more
# The scanner rides about 6 mm low on the GPU (the cup's centre 164.5 mm up for the 170 mm line; 8 mm is the limit):
# the slide is commanded this much above the line.
LINE_DZ = 0.004
APPROACH = 0.030  # the cup 30 mm off the skin before it moves in
PHI_END = math.radians(41.5)  # the cup's centre commanded to 54 mm round the curve (its edge 2 mm short of the hair)
SWEEP_S = 3.0
SWEEP_SEGMENTS = 6  # 3 chords sagged off the curve between waypoints on the GPU
PICK_Z = L.PR_Z0
PLACE_Z = PICK_Z - 0.001
# Carrying the 213 g scanner the arm sat 4 to 5 mm short in x (GPU job 20261120T075550: TCP x 219.9 mm for 224.7
# commanded), so the body came down on the foam blocks' tops (12 mm high, 3.5 mm clearance each side in the 34 mm slots)
# and toppled when let go. The put-back leads by this much in x.
PLACE_DX = 0.0045
HOVER_Z = 0.110
PARK_R = (0.14, -0.34, 0.180)


def gripper_yaw(psi: float) -> float:
    return psi + math.pi / 2


def tcp_for(lens_xy, z: float, psi: float, grip=L.GRIP_LOCAL) -> tuple[float, float, float]:
    """The TCP that puts the cup's centre at ``lens_xy`` (height ``z``) with the body frame turned ``psi``."""
    g = rot_z(psi) @ np.asarray(grip, dtype=float)
    return float(lens_xy[0] + g[0]), float(lens_xy[1] + g[1]), float(z + g[2])


def _w(d, p, finger, label, psi):
    return Waypoint(d, (p[0], p[1], L.TOP + p[2]), finger, label, gripper_yaw(psi))


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def scan_plan(pen=PEN, sweep_s=SWEEP_S, follow=True, dz=0.0, grip=L.GRIP_LOCAL, put_back=True):
    """The right arm's plan. ``psi`` below is the body's yaw; with the cup facing the skin's normal at angle phi round
    the curve, the body's yaw is phi - CUP_TILT (the cup faces 30 degrees off the body's back)."""
    z = L.Z_LINE + LINE_DZ + dz
    rest = (L.PR_X0, L.PR_Y0)
    pick = tcp_for(rest, PICK_Z, 0.0, grip)
    b0 = -L.CUP_TILT  # body yaw with the cup facing +x (the forehead centre)
    start_out = L.arc_point(0.0, L.RF + APPROACH)
    centre = L.arc_point(0.0, L.RF - pen)
    plan = [
        _w(2.0, (pick[0], pick[1], HOVER_Z), OPEN_PR, "over the thermometer", 0.0),
        _w(1.0, (pick[0], pick[1], PICK_Z + 0.025), OPEN_PR, "down to just above the rest", 0.0),
        _w(1.5, pick, OPEN_PR, "slowly down beside it", 0.0),
        _w(0.5, pick, OPEN_PR, "settle", 0.0),
        _w(1.5, pick, GRIP_PR, "grip it across the SCAN button and press it", 0.0),
        _w(1.0, (pick[0], pick[1], PICK_Z + 0.035), GRIP_PR, "lift it out of the slots", 0.0),
        _w(2.5, tcp_for(start_out, z, b0, grip), GRIP_PR, "carry it in front of the forehead centre", b0),
        _w(2.0, tcp_for(centre, z, b0, grip), GRIP_PR, "cup flush on the forehead centre", b0),
        _w(0.8, tcp_for(centre, z, b0, grip), GRIP_PR, "settle on the centre", b0),
    ]
    for k in range(1, SWEEP_SEGMENTS + 1):
        phi = PHI_END * k / SWEEP_SEGMENTS
        psi = b0 + (phi if follow else 0.0)
        plan.append(_w(sweep_s / SWEEP_SEGMENTS, tcp_for(L.arc_point(L.RF * phi, L.RF - pen), z, psi, grip), GRIP_PR,
                       f"slide round the forehead {k}/{SWEEP_SEGMENTS}", psi))
    psi_end = b0 + (PHI_END if follow else 0.0)
    end_out = L.arc_point(L.RF * PHI_END, L.RF + APPROACH)
    away = (end_out[0] - 0.040, end_out[1])
    plan += [
        _w(0.3, tcp_for(L.arc_point(L.RF * PHI_END, L.RF - pen), z, psi_end, grip), GRIP_PR, "hold at the hairline", psi_end),
        _w(1.0, tcp_for(end_out, z, psi_end, grip), GRIP_PR, "lift the cup off the skin", psi_end),
        _w(1.0, tcp_for(away, z, psi_end, grip), GRIP_PR, "back away from the head", psi_end),
    ]
    if not put_back:
        plan.append(_w(6.0, tcp_for(away, z, psi_end, grip), GRIP_PR, "keep holding it", psi_end))
        return plan
    plan += [
        _w(2.5, (pick[0] + PLACE_DX, pick[1], HOVER_Z), GRIP_PR, "carry it back over the rest", 0.0),
        _w(1.5, (pick[0] + PLACE_DX, pick[1], PICK_Z + 0.025), GRIP_PR, "down to just above the rest", 0.0),
        _w(2.0, (pick[0] + PLACE_DX, pick[1], PLACE_Z), GRIP_PR, "slowly lower it into the slots", 0.0),
        _w(0.8, (pick[0] + PLACE_DX, pick[1], PLACE_Z), GRIP_PR, "settle on the slot floors", 0.0),
        _w(1.5, (pick[0] + PLACE_DX, pick[1], PLACE_Z), OPEN_PR, "let go (the button comes out)", 0.0),
        _w(1.0, (pick[0] + PLACE_DX, pick[1], PLACE_Z), OPEN_PR, "hand open, hold still", 0.0),
        _w(1.5, (pick[0] + PLACE_DX, pick[1], HOVER_Z), OPEN_PR, "straight up", 0.0),
        _w(2.0, PARK_R, OPEN, "aside", 0.0),
        _w(4.0, PARK_R, OPEN, "aside: hands off, rest", 0.0),
    ]
    return plan


# Wrong ways (counterfactuals, for the README; never the submitted demo):
#   no_button    gripped 40 mm further down the body, on the display, so the SCAN button is never pressed
#   fast         the slide in 0.9 s (a flick)
#   light        the cup commanded only 1 mm into the skin
#   hard         the cup commanded 18 mm into the skin (the head shoved back)
#   no_follow    the slide round the curve with the scanner never turned: the cup is not flush past the centre
#   low          the slide 12 mm below the scan line (toward the brows)
#   keep_probe   the thermometer is never put back
VARIANTS = ("no_button", "fast", "light", "hard", "no_follow", "low", "keep_probe")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    kw = {}
    if variant == "no_button":
        kw["grip"] = (L.GRIP_LOCAL[0], L.BTN_Y - 0.040, 0.0)
    elif variant == "fast":
        kw["sweep_s"] = 0.9
    elif variant == "light":
        kw["pen"] = 0.001
    elif variant == "hard":
        kw["pen"] = 0.018
    elif variant == "no_follow":
        kw["follow"] = False
    elif variant == "low":
        kw["dz"] = -0.012
    elif variant == "keep_probe":
        kw["put_back"] = False
    return {"right": scan_plan(**kw)}


def demo():
    """The right arm's plan (the left holds its ready pose). DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_IDLE_ARM=right drops the plan. DEMO_VARIANT: see VARIANTS (never the submitted demo)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
