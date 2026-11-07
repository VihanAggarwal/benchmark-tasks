"""On the station table: a fixed aluminium air track (30 mm wide, 25 mm tall, 440 mm long) runs along y on the
midline, 28 cm in front of the arm bases, with a black rubber buffer at each end and a blue glider parked (air off)
against the left (+y) buffer, a black interrupt card standing on it. Two loose black light gates (inverted U frames
on two feet, 92 mm across, 84 mm tall, a red sensor window on each leg's inner face) stand on the bench, one beside
each arm, turned along the track's direction. Rule: set each gate over the track, straddling it, square to it, with both
feet on the bench, in front of the glider, the two gates 200 mm apart (centre to centre, along the track). A gate
resting on the track, or anywhere in the glider's path, would stop the run: 0.

Each gate is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of seven boxes, so it
stands on its feet over the track and the scorer can read its pose (base frame and orientation).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the gates are built as compound bodies)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The second gate can come to rest on the bench while the slow descent is
# still finishing (the arm sags a few mm under it), and success starts counting 0.5 s later (the time needed to see
# it at rest) with the gripper still closed. Settling and opening take 0.5 + 1.5 s, so with 2.5 s the trial ends with
# the fingers open and clear of the crossbar. Offline (no sag) the gate is let go with its feet 2 mm above the bench.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Set the two light gates over the air track for the speed run: each one straddling the track with both feet "
    "on the bench and nothing of it in the glider's path, square to the track, in front of the parked glider, the "
    "two gates 200 mm apart, centre to centre."
)


def _box(name: str, x: float, y: float, z0: float, size, color) -> Box:
    """A fixed box by its centre (x, y), its bottom height above the bench and its size."""
    return Box(name, pos=(x, y, L.TOP + z0 + size[2] / 2), size=size, color=color)


TRACK = _box("air_track", L.TRACK_X, 0.0, 0.0, (L.TRACK_W, L.TRACK_LEN, L.TRACK_H), L.ALUMINIUM)
BUFFERS = [
    _box("buffer_start", L.TRACK_X, L.BUFFER_Y, 0.0, L.BUFFER_SIZE, L.RUBBER_BLACK),
    _box("buffer_end", L.TRACK_X, -L.BUFFER_Y, 0.0, L.BUFFER_SIZE, L.RUBBER_BLACK),
]
GLIDER = [
    _box("glider_plate", L.TRACK_X, L.GLIDER_Y, L.TRACK_H, L.GLIDER_PLATE, L.GLIDER_BLUE),
    _box("glider_skirt_front", L.TRACK_X + L.GLIDER_SKIRT_X, L.GLIDER_Y, L.GLIDER_SKIRT_Z0, L.GLIDER_SKIRT,
         L.GLIDER_BLUE),
    _box("glider_skirt_back", L.TRACK_X - L.GLIDER_SKIRT_X, L.GLIDER_Y, L.GLIDER_SKIRT_Z0, L.GLIDER_SKIRT,
         L.GLIDER_BLUE),
    _box("glider_card", L.TRACK_X, L.GLIDER_Y, L.TRACK_H + L.GLIDER_PLATE[2], L.GLIDER_CARD, L.CARD_BLACK),
]


def _gate(name: str):
    """A light gate: a loose inverted U standing on its two feet, base frame at the centre between them."""
    sx, sy = L.START[name]
    shapes = [mk.box(c, s, color=L.SENSOR_RED if s == L.WINDOW_SIZE else None) for c, s in L.GATE_BOXES]
    return mk.assembly(name, origin=(sx, sy, L.TOP), yaw=L.START_YAW, mass=L.GATE_MASS, color=L.GATE_BLACK,
                       shapes=shapes)


GATE_LEFT = _gate("gate_left")
GATE_RIGHT = _gate("gate_right")
OBJECTS = [TRACK, *BUFFERS, *GLIDER, GATE_LEFT, GATE_RIGHT]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


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
    """Both gates set (standing on the bench straddling the track, square within 10 degrees, centre in the timing
    section, at rest), centres 190-210 mm apart along the track, nothing of either gate in the glider's path; held
    2.5 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    gates = {n: objects[n] for n in L.GATES}
    result = _TRACKER.update(gates)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, g in gates.items():
        try:
            row[n + "_state"] = {"source": getattr(g, "source", None), "origin": getattr(g, "origin", None),
                                 "quat": g.quat_xyzw, "vel": g.lin_vel}
        except AttributeError:
            row[n + "_state"] = "unreadable"
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any part of either gate is in the glider's path (within 22 mm of the track's centre line, lower than 57 mm,
    between the buffers). Otherwise 0.25 for each set gate, plus, when both are set, 0.5 for centres 190-210 mm apart
    along the track or 0.25 for 180-220 mm. Read from the state at hand (the last step)."""
    return _TRACKER.result({n: objects[n] for n in L.GATES}).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
GATE_OPEN = -0.030  # 60 mm: 20 mm either side of the crossbar's 20 mm depth
GATE_GRIP = -(L.CROSSBAR_SIZE[1] - 0.008) / 2  # 12 mm commanded across the 20 mm crossbar: about 20 N per finger
# Each gate is picked by the middle of its crossbar with the fingers closing across it (the gate starts turned along
# y, so they close along x: the left arm's yaw -pi/2, the right arm's +pi/2), turned a quarter turn in the hand while
# it is carried, and set down square to the track with the fingers closing along y (yaw 0 for both arms).
YAW_PICK = {"left": -math.pi / 2, "right": math.pi / 2}
YAW_PLACE = 0.0
GRIP_UP = 0.082  # the TCP 82 mm above the gate's feet: pad tips 69 mm up, 5 mm below the crossbar's underside
DROP = 0.002  # the gate is let go with its feet 2 mm above the bench
APPROACH = 0.030  # each final descent starts this far above the grip height
# Carried with its feet 58 mm up: 33 mm over the track's top (25 mm) and 23 mm over the buffers (35 mm). Each gate
# goes first straight along y to x 0.20 (VIA_X), turning square on the way, then 80 mm across onto the track, so it
# never passes over the glider: carried straight from its start to the track, the left gate's feet passed 3 mm over
# the glider's card (55 mm up; offline replay of the gate's seven boxes), and carrying higher costs joint 4 its
# margin (10 deg at 0.17 m).
CARRY_Z = L.TOP + 0.14
PARK_Z = L.TOP + 0.15
VIA_X = 0.200  # turning here, the gate's corners (50 mm from its centre) reach x 0.250: 9 mm short of the glider
# Where each gate is set: centred on the track, 100 mm either side of the midline (200 mm apart), both in the timing
# section. Trim (x, y), for the GPU run: if a gate lands off its aim (centre_mm in TASK_TRACE away from
# (280, +-100), or path_gap_mm small), move its place point by minus the offset. The light gate (80 g) loads each
# arm far less than the 0.2-0.3 kg parts of the earlier runs, so the expected offset is a millimetre or two.
PLACE_Y = {"left": L.SPACING_AIM / 2, "right": -L.SPACING_AIM / 2}
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
# The right arm waits over its gate so that it carries its gate toward the track only once the left arm has let go,
# risen and started back: both place points are on x 0.28, 200 mm apart, the minimum spacing for two grippers at one
# x. Offline (capsule proxies) the arms' links then stay at least 14.8 cm apart; with a 9 s wait the two wrists came
# within 4.7 cm (the left going back while the right carried its gate in front of the track).
RIGHT_WAIT = 10.5


def _arm(side: str, wait: float = 0.0) -> list[Waypoint]:
    """One arm: over its gate, grip the crossbar, lift, carry it in front of the track turning it square, across over
    the track, lower it slowly astride the track, let go, go up and back over its starting place."""
    gate = "gate_left" if side == "left" else "gate_right"
    px, py = L.START[gate]
    pz = L.TOP + GRIP_UP
    qx, qy = L.TRACK_X + PLACE_TRIM[side][0], PLACE_Y[side] + PLACE_TRIM[side][1]
    qz = L.TOP + DROP + GRIP_UP
    y0 = YAW_PICK[side]
    plan = [Waypoint(2.0, (px, py, CARRY_Z), GATE_OPEN, "over the light gate", y0)]
    if wait > 0:
        plan.append(Waypoint(wait, (px, py, CARRY_Z), GATE_OPEN, "wait for the other arm to set its gate", y0))
    plan += [
        Waypoint(1.0, (px, py, pz + APPROACH), GATE_OPEN, "down to just above the crossbar", y0),
        Waypoint(1.5, (px, py, pz), GATE_OPEN, "slowly down round the crossbar", y0),
        Waypoint(0.5, (px, py, pz), GATE_OPEN, "settle", y0),
        Waypoint(1.5, (px, py, pz), GATE_GRIP, "grip the crossbar", y0),
        Waypoint(0.3, (px, py, pz), GATE_GRIP, "settle the grip", y0),
        Waypoint(1.5, (px, py, CARRY_Z), GATE_GRIP, "lift the gate clear of the bench", y0),
        Waypoint(2.5, (VIA_X, qy, CARRY_Z), GATE_GRIP, "carry it in front of the track, turning it square",
                 YAW_PLACE),
        Waypoint(1.5, (qx, qy, CARRY_Z), GATE_GRIP, "carry it across over the track", YAW_PLACE),
        Waypoint(0.8, (qx, qy, CARRY_Z), GATE_GRIP, "settle over the track", YAW_PLACE),
        Waypoint(1.0, (qx, qy, qz + APPROACH), GATE_GRIP, "down until the feet are 32 mm up", YAW_PLACE),
        Waypoint(2.0, (qx, qy, qz), GATE_GRIP, "slowly lower it astride the track", YAW_PLACE),
        Waypoint(0.5, (qx, qy, qz), GATE_GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (qx, qy, qz), GATE_OPEN, "let go of the gate", YAW_PLACE),
        Waypoint(0.5, (qx, qy, qz), GATE_OPEN, "settle", YAW_PLACE),
        Waypoint(1.0, (qx, qy, CARRY_Z), GATE_OPEN, "straight up", YAW_PLACE),
        Waypoint(1.5, (px, py, PARK_Z), GATE_OPEN, "back over the starting place", YAW_PLACE),
    ]
    return plan


def demo():
    """Both arms on one clock: each sets the gate on its own side, the right arm coming down over the track only
    after the left has let go and risen. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right", RIGHT_WAIT)}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
