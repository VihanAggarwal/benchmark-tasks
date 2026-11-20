"""On the station table, real size: a low media-style cabinet (laminated chipboard, open front at x 360 mm, 620 mm wide,
270 mm clear between its floor and the underside of its top track). Under the top board a double aluminium-style top
track holds two grooves, 38 mm wide and 40 mm deep; on the floor (40 mm up) two rails, 6 mm wide and 6 mm high, run
under them. The rear door (22 mm honeycomb panel, 320 x 290 mm, 0.29 kg, a 14 x 8 mm groove in its foot) is already
hung in the rear track over y -300 to +20 mm. The identical front door, with a 140 mm bar pull standing 30 mm off its
face, stands on the table at x 337 mm, face toward the robot.

Goal: the front door hung in the front track (top 20 mm up in the groove, foot on the floor with the rail in its
groove), the rear door where it was, both hands off. The door is 290 mm tall and the opening 270 mm clear, so it
cannot go in upright: its top has to be leaned in under the front groove, lifted up into it, its foot swung in over
the floor's edge while the top stays in the groove, and the door lowered onto the rail.

The parts are mechkit compounds (mechkit.py, next to this file). No declared software rule: everything is rigid bodies
and contacts. The arms are read through rigstate.py (the let-go check). tiltgrip.py pitches the demo's grip.

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

import geom as G  # noqa: E402
import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (pitched grips for the demo: wraps yam_arena.demo.grip_rotation)
from scoring import FRONT, REAR  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "Hang the loose white door, still facing the robot, in the cabinet's front track and let go of it, leaving the "
    "rear door where it is."
)


# -- the parts --------------------------------------------------------------------------------------------------------
def _b(x0, x1, y0, y1, z0, z1, color=L.CAB_WOOD):
    """A box from its extents (x, y, z)."""
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def cabinet_shapes():
    """All in the world frame with z from the table top (the assembly's origin is (0, 0, table top))."""
    f, r = L.FRONT_GROOVE, L.REAR_GROOVE
    y_in, y_out = L.OPEN_HALF, L.OPEN_HALF + L.BOARD
    x0, x1 = L.CAB_X0, L.CAB_X1
    out = [
        _b(x0, x1, -y_out, y_out, 0.0, L.FLOOR_TOP),  # the plinth and floor board
        _b(x0, x1, -y_out, -y_in, L.FLOOR_TOP, L.TOP_PANEL_TOP),  # right side panel
        _b(x0, x1, y_in, y_out, L.FLOOR_TOP, L.TOP_PANEL_TOP),  # left side panel
        _b(x1 - L.BOARD, x1, -y_in, y_in, L.FLOOR_TOP, L.ROOF),  # back panel
        _b(x0, x1, -y_in, y_in, L.ROOF, L.TOP_PANEL_TOP),  # top board
        _b(L.TRACK_X0, f[0], -y_in, y_in, L.TRACK_BOTTOM, L.ROOF, L.ALUMINIUM),  # the top track's walls
        _b(f[1], r[0], -y_in, y_in, L.TRACK_BOTTOM, L.ROOF, L.ALUMINIUM),
        _b(r[1], L.TRACK_X1, -y_in, y_in, L.TRACK_BOTTOM, L.ROOF, L.ALUMINIUM),
    ]
    for xc in (L.X_F, L.X_R):  # the bottom rails
        out.append(_b(xc - L.RAIL_W / 2, xc + L.RAIL_W / 2, -y_in, y_in, L.FLOOR_TOP, L.RAIL_TOP, L.ALUMINIUM))
    return out


def door_shapes(front: bool):
    """Base frame: the middle of the underside; x across the thickness, y along the width, z up."""
    t, w, h, g, d = L.DOOR_T, L.DOOR_W, L.DOOR_H, L.FOOT_GROOVE_W / 2, L.FOOT_GROOVE_D
    out = [mk.box((0.0, 0.0, (d + h) / 2), (t, w, h - d), L.DOOR_WHITE)]
    for s in (-1, 1):  # the strips either side of the foot groove
        out.append(mk.box((s * (g + L.STRIP / 2), 0.0, d / 2), (L.STRIP, w, d), L.DOOR_WHITE))
    if front:
        out.append(mk.box((L.PULL_X, 0.0, L.PULL_H), L.PULL_SIZE, L.PULL_STEEL))
        px = -(t / 2 + L.PULL_OFF / 2)
        for s in (-1, 1):
            out.append(mk.box((px, s * L.POST_Y, L.PULL_H), L.POST, L.PULL_STEEL))
    return out


CABINET = mk.assembly("cabinet", origin=(0.0, 0.0, L.TOP), shapes=cabinet_shapes(), color=L.CAB_WOOD,
                      friction=L.FRICTION, tight_fit=True)
FRONT_PART = mk.assembly(FRONT, origin=(L.FRONT_START_X, L.FRONT_START_Y, L.TOP), mass=L.DOOR_MASS,
                         shapes=door_shapes(True), color=L.DOOR_WHITE, friction=L.FRICTION, tight_fit=True)
REAR_PART = mk.assembly(REAR, origin=(L.REAR_X, L.REAR_Y, L.TOP + L.FLOOR_TOP), mass=L.DOOR_MASS,
                        shapes=door_shapes(False), color=L.DOOR_WHITE, friction=L.FRICTION, tight_fit=True)
OBJECTS = [CABINET, FRONT_PART, REAR_PART]
SPECS = {FRONT: FRONT_PART, REAR: REAR_PART}


# -- the offline stand-in for the physics ---------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it knows nothing of a door that leans
# or of a groove. Offline only (never in the simulator), this model gives the scorer: (1) the front door rides on the
# left gripper once its pads are closed on the bar (hand within 12 mm of the bar, its lean within 12 degrees of the
# hand's pitch), the lean taken from the hand's approach axis and the pose resolved against the cabinet and the rear
# door in the x-z plane (a door that would pass through them stops at the contact, the hand going on); (2) a released
# door falls straight down to the first contact, then, with its top inside the front groove, swings upright about its
# top (the groove's walls hold the top, the foot slides to under it) and drops again; a door whose top is not in
# the groove and that leans more than 3 degrees falls over (lies on its face).
GRIP_NEAR = 0.012  # the hand within this of the bar's centre to take hold
GRIP_LEAN_DEG = 12.0
OPEN_AT = 0.6  # opening above this lets go
FALL_OVER_DEG = 3.0


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.b = np.array([L.FRONT_START_X, 0.0])  # the front door's underside centre in the plane (x, z)
        self.theta = 0.0
        self.y = L.FRONT_START_Y
        self.held = False
        self.hold_dy = 0.0
        self.max_pen = 0.0  # the deepest the hand pushed the door into the cabinet (m)
        self.lying = False
        self.arm = None

    @staticmethod
    def _approach(arm):
        a = np.asarray(arm.tips, dtype=float) - np.asarray(arm.tcp, dtype=float)
        n = float(np.linalg.norm(a))
        return a / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _resolve(self, b_new, th_new):
        """Move to the new pose, sweeping from the old one in 1 mm steps, and stop at the first contact with the
        cabinet or the rear door (the hand goes on: the door stays at the contact)."""
        b_new, b0, t0 = np.asarray(b_new, dtype=float), self.b.copy(), self.theta
        dist = max(float(np.hypot(*(b_new - b0))), abs(th_new - t0) * 0.3)
        n = max(1, int(math.ceil(dist / 0.001)))
        for i in range(1, n + 1):
            f = i / n
            bb, tt = b0 + f * (b_new - b0), t0 + f * (th_new - t0)
            if G.clearance(bb, tt) < -1e-4:
                self.max_pen = max(self.max_pen, -G.clearance(b_new, th_new))
                return
            self.b, self.theta = bb, tt

    def _fall(self) -> None:
        """Straight down to the first contact."""
        step = 0.0005
        for _ in range(200):
            nb = self.b + np.array([0.0, -step])
            if G.clearance(nb, self.theta) < 0.0:
                return
            self.b = nb

    def _settle(self) -> None:
        """A released door: fall, swing upright if its top is in the groove, fall again; or topple."""
        self._fall()
        tx0, tx1, tz = G.top_corners_x(self.b, self.theta)
        lo, hi = L.FRONT_GROOVE
        in_groove = lo - 0.002 <= tx0 and tx1 <= hi + 0.002 and tz >= L.TRACK_BOTTOM
        if in_groove and abs(self.theta) < math.radians(15.0):
            if self.theta > 0.0 and G.clearance(self.b, self.theta) < 0.002:
                # on the floor or rail beside the track: the foot cannot slide in while it rests there
                return
            top = 0.5 * (tx0 + tx1)
            self.theta = 0.0
            self.b = np.array([top, self.b[1]])
            self._fall()
            return
        if abs(self.theta) > math.radians(FALL_OVER_DEG) and not in_groove:
            self.lying = True

    def step(self, arms) -> None:
        arm = None if arms is None else arms.get("left")
        if arm is None:
            return
        a = self._approach(arm)
        lean = G.lean_from_approach((a[0], a[2]))
        tcp = np.asarray(arm.tcp, dtype=float)
        p, lean = sagged((float(tcp[0]), float(tcp[1]), float(tcp[2])), lean)  # the arm as the GPU measured it
        tcp = np.array(p)
        tcp_xz = np.array([tcp[0], tcp[2] - L.TOP])
        bar = G.tcp_for(self.b, self.theta)
        if not self.held and not self.lying:
            near = float(np.hypot(*(tcp_xz - bar))) <= GRIP_NEAR and abs(tcp[1] - self.y) <= 0.03
            aligned = abs(math.degrees(lean - self.theta)) <= GRIP_LEAN_DEG and abs(a[1]) < 0.3
            if arm.opening < OPEN_AT and near and aligned:
                self.held = True
                self.hold_dy = self.y - float(tcp[1])
        elif self.held and (arm.opening >= OPEN_AT or self.lying):
            self.held = False
            self._settle()
        if self.held:
            self._resolve(G.pose_for_tcp(tcp_xz, lean), lean)
            self.y = float(tcp[1]) + self.hold_dy

    def states(self) -> dict:
        if self.lying:
            q = mk.quat_from_ypr(0.0, math.pi / 2, 0.0)
            origin = (float(self.b[0]), self.y, L.TOP + 0.011)
        else:
            q = mk.quat_from_ypr(0.0, self.theta, 0.0)
            origin = (float(self.b[0]), self.y, L.TOP + float(self.b[1]))
        fr = mk.state_from_poses(FRONT_PART, (*origin, *q), source="offline")
        rr = mk.state_from_poses(REAR_PART, (L.REAR_X, L.REAR_Y, L.TOP + L.FLOOR_TOP, 0.0, 0.0, 0.0, 1.0),
                                 source="offline")
        return {FRONT: fr, REAR: rr}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(FRONT), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, top min z mm, foot height mm, tilt deg, face mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: both doors' poses and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The front door hung (top up in the front groove, foot on the floor over the rail, upright), the rear door
    kept, both grippers 30 mm clear and both doors still for 1 s; held 1 s. This runs first every control step, so it
    also feeds the tracker (and, without a simulator, the offline stand-in)."""
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
    row["source"] = getattr(parts.get(FRONT), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["top_min_z_mm"], d["foot_height_mm"], d["tilt_deg"], d["face_offset_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the rear door was knocked out (latched), or unless both doors are let go and still; 1.0 the front door
    hung (top 6 mm or more up in the front groove, foot within 3 mm of the floor, face within 7 mm of the track,
    under 3 degrees), rear door kept; 0.5 top in the groove and foot down, tilted under 8 degrees; 0.25 top in the
    groove otherwise, or the rear door moved; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. The TCP is at the bar's
# centre. The grip is closed along the door's up axis (pads above and below the 20 mm bar), the approach axis
# pointing at the door: horizontal at the start (pitch 90 deg), leaning with the door afterwards.
D = math.radians
OPEN = -0.030  # 60 mm: round the 20 mm bar with 20 mm each side
GRIP = -(L.PULL_SIZE[2] - 0.014) / 2  # 6 mm commanded across the 20 mm bar: 7 mm short each side, about 35 N a pad:
#                                       the door's weight acts 5 cm out from the pads and must not turn it
YAW = -math.pi / 2  # the closing axis along world x before the pitch: along the door's up axis after it
PY = L.FRONT_START_Y  # the grip's y at the start: the middle of the door
PY_HUNG = L.FRONT_HUNG_Y  # where the door ends: 120 mm further left, under the groove
LIFT0 = 0.012  # the door off the table (commanded through the sag lead below)
LEAN = 10.0  # degrees: the top leaned in under the front groove
TOP_UNDER = L.TRACK_BOTTOM - 0.007  # 303 mm: the top's height, 7 mm under the groove's mouth, while it comes in
# keyframes of the door's top-edge centre (x, z, lean in degrees): lean in, lift into the groove while turning upright
KEYS = [
    ("lean", (L.X_F, TOP_UNDER, LEAN)),
    ("rise", (L.X_F, 0.332, LEAN)),  # 7 mm higher than first planned: the foot clears the floor's edge by 7 mm
    ("swing 7", (L.X_F, 0.342, 7.0)),
    ("swing 3.5", (L.X_F, 0.346, 3.5)),
    ("upright, up to the roof", (L.X_F, L.ROOF + 0.004, 0.0)),  # 4 mm past the roof: the arm sags more than that
]
# The loaded arm, measured on the GPU (job 20261117T001944, left arm carrying the door by the bar): it does not reach
# what it is told. Commanded TCP z c against reached z: 0.918 -> 0.918, 0.930 -> 0.920, 0.941 -> 0.928, 0.963 -> 0.942,
# 0.973 -> 0.949, 0.982 -> 0.949; past 0.930 the error is -(9.7 mm + 0.36 per mm of c over 0.930). The door's tilt
# read 2.4 to 4.5 degrees more than the hand's pitch at the same time (the wrist gives under the door's 0.14 N m).
# The demo leads the sag (the commanded z is raised, the hand's pitch lowered) while the hand carries the door in
# the air; the offline stand-in applies the same sag to what it carries, so the offline check sees the real arm.
SAG_FROM = 0.918  # no error at or below this commanded z (the hand at the bar, door on the table)
SAG_AT = 0.930
SAG_Z0 = 0.0097
SAG_SLOPE = 0.36
SAG_PITCH = 3.0  # degrees: the hand's commanded pitch is this far under the door's wanted lean


def sag_z(c: float) -> float:
    """Reached minus commanded TCP z (m) of the arm carrying the door, for a commanded z c."""
    if c <= SAG_FROM:
        return 0.0
    if c <= SAG_AT:
        return -SAG_Z0 * (c - SAG_FROM) / (SAG_AT - SAG_FROM)
    return -(SAG_Z0 + SAG_SLOPE * (c - SAG_AT))


def lead_z(a: float) -> float:
    """The commanded z that the loaded arm reaches as ``a`` (the inverse of c + sag_z(c))."""
    if a <= SAG_FROM:
        return a
    a_at = SAG_AT + sag_z(SAG_AT)
    if a <= a_at:
        return SAG_FROM + (a - SAG_FROM) * (SAG_AT - SAG_FROM) / (a_at - SAG_FROM)
    return SAG_AT + (a - a_at) / (1.0 - SAG_SLOPE)


def sagged(pos, lean: float):
    """What the loaded arm reaches for a commanded TCP ``pos`` and door lean (rad), once it carries the door in the
    air (commanded z over SAG_FROM): (pos, lean) as reached."""
    if pos[2] <= SAG_FROM + 1e-9:
        return pos, lean
    return (pos[0], pos[1], pos[2] + sag_z(pos[2])), lean + math.radians(SAG_PITCH)


LOWERED_Z = L.FLOOR_TOP + 0.004  # the foot commanded 4 mm over the floor: the sag puts it on it
RETREAT = 0.130  # the hand backs off this far toward the robot after letting go (its fingers 30 mm clear)
PRE = 0.060  # the open hand starts this far in front of the bar
HOVER = 0.050  # and goes up this far when it has backed off
VARIANTS = ("short", "out")  # short: let go leaning 3.5 degrees, foot 16 mm up, 17 mm in front of the rail (lands
#   on the floor in front of the rail, still leaning: 0.5); out: let go leaning 10 degrees, top just in the groove
#   (falls out: 0)


def _tcp(b, theta, y=PY):
    xz = G.tcp_for(b, theta)
    return (float(xz[0]), y, L.TOP + float(xz[1]))


def _w(d, pos, finger, label, lean_deg, carried=False, holding=False):
    """A waypoint. ``carried``: the hand holds the door in the air, so lead the arm's sag (z raised, pitch lowered).
    ``holding``: the pitch only (the door is on the floor or about to be: the z is left as it is)."""
    if carried:
        pos = (pos[0], pos[1], lead_z(pos[2]))
    if carried or holding:
        lean_deg = lean_deg - SAG_PITCH
    return Waypoint(d, pos, finger, label, tiltgrip.tilted(YAW, G.pitch_for(D(lean_deg))))


def _key(name):
    top, lean = dict(KEYS)[name][:2], dict(KEYS)[name][2]
    return G.top_for(top, D(lean)), lean


def plan(variant: str = ""):
    """The left arm's waypoints (the right arm has none: it holds its ready pose)."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    start_b = (L.FRONT_START_X, 0.0)
    bar0 = _tcp(start_b, 0.0)
    out = [
        _w(2.5, (bar0[0] - PRE, PY, bar0[2]), OPEN, "in front of the bar, hand level", 0.0),
        _w(1.5, bar0, OPEN, "pads over and under the bar", 0.0),
        _w(0.5, bar0, OPEN, "settle", 0.0),
        _w(1.5, bar0, GRIP, "close on the bar", 0.0),
        _w(0.5, bar0, GRIP, "settle the grip", 0.0),
        _w(1.5, _tcp((L.FRONT_START_X, LIFT0), 0.0), GRIP, "lift the door off the table", 0.0, carried=True),
    ]
    wp_y = PY_HUNG
    # 1) slide the lifted door level along y under the groove (door 337 mm, rear face 12 mm from the floor board's
    #    front face: nothing moves in x).  2) lean it about its foot in four legs: the door's own pose is stepped, so
    #    the foot stays at x 337 (a straight line between the hand poses swung it 10 mm toward the floor board in
    #    the GPU run: the rear-bottom corner met the board's front face and the door stood vertical against it).
    out.append(_w(1.8, _tcp((L.FRONT_START_X, LIFT0), 0.0, wp_y), GRIP, "slide the lifted door along y", 0.0,
                  carried=True))
    lean_b, lean_deg = _key("lean")
    for f in (0.25, 0.5, 0.75, 1.0):
        b = np.array([L.FRONT_START_X, LIFT0]) + f * (np.asarray(lean_b) - np.array([L.FRONT_START_X, LIFT0]))
        out.append(_w(0.7 if f < 1.0 else 0.8, _tcp(b, D(f * lean_deg), wp_y), GRIP,
                      f"lean the door about its foot, {f * lean_deg:.1f} degrees", f * lean_deg, carried=True))
    out.append(_w(0.7, _tcp(lean_b, D(lean_deg), wp_y), GRIP, "settle the lean", lean_deg, carried=True))
    names = ["rise", "swing 7", "swing 3.5", "upright, up to the roof"]
    times = {"rise": 1.5, "swing 7": 1.5, "swing 3.5": 1.5, "upright, up to the roof": 1.5}
    labels = {"lean": "lean its top in under the groove, 10 degrees", "rise": "up into the groove, still leaning",
              "swing 7": "up and swing the foot in, 7 degrees", "swing 3.5": "up and swing the foot in, 3.5 degrees",
              "upright, up to the roof": "upright, top at the groove's roof"}
    stop = {"short": "swing 3.5", "out": "rise"}.get(variant)
    for n in names:
        b, lean = _key(n)
        out.append(_w(times[n], _tcp(b, D(lean), wp_y), GRIP, labels[n], lean, carried=True))
        if n == stop:
            out.append(_w(0.5, _tcp(b, D(lean), wp_y), GRIP, "settle", lean, carried=True))
            out.append(_w(1.5, _tcp(b, D(lean), wp_y), OPEN, "let go", lean, carried=True))
            held = _tcp(b, D(lean), wp_y)
            held = (held[0], held[1], lead_z(held[2]))
            out.append(_w(1.5, (held[0] - RETREAT, held[1], held[2]), OPEN, "back off", lean))
            out.append(_w(1.5, (held[0] - RETREAT, held[1], held[2] + HOVER), OPEN, "up", 0.0))
            return out
    top = _tcp((L.X_F, LOWERED_Z), 0.0, wp_y)
    out += [
        _w(0.5, _tcp(_key("upright, up to the roof")[0], 0.0, wp_y), GRIP, "settle against the roof", 0.0,
           carried=True),
        _w(2.0, top, GRIP, "lower it onto the rail", 0.0, holding=True),
        _w(0.5, top, GRIP, "settle", 0.0, holding=True),
        _w(1.5, top, OPEN, "let go", 0.0, holding=True),
        _w(1.5, (top[0] - RETREAT, top[1], top[2]), OPEN, "back off toward the robot", 0.0),
        _w(1.5, (top[0] - RETREAT, top[1], top[2] + HOVER), OPEN, "up", 0.0),
    ]
    return out


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo): short, out (see VARIANTS)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plans = {"left": plan(os.environ.get("DEMO_VARIANT", ""))}
    plans.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plans
