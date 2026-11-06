"""On the station table: a fixed child's snowboard top (1100 x 240 x 10 mm) lying across the table along y, nose to the
left, heel edge 18 cm in front of the arm bases: two white strips with a 4 mm black centre line between them at x 30 cm,
a small red reference cross at its middle (y 0) and two grey insert grids (2 x 4 dot stickers on a 40 mm pitch)
centred at y +20 and -20 cm. Two loose bindings (0.25 kg each: baseplate, disc, highback and an ankle-strap bar) lie
parked flat on the board's two ends, discs on the centre line at y +-44.5 cm, toes toward the ends. Rule: set a duck
stance: each binding flat with its disc over a grid (|y| 14 to 26 cm) and centred on the line, the discs 40 cm apart
and centred on the cross, the binding nearer the nose turned +15 deg (toe toward the nose), the other -15 deg (toe
toward the tail), both let go with the grippers lifted clear.

The bindings are mechkit assemblies (mechkit.py, next to this file: each a loose compound rigid body of five shapes);
the scorer reads each one's frame from it, and the arms (TCP, finger-tip point, opening) through rigstate.py, which
wraps the embodiment's step and reset from this file's import. Neither changes what the embodiment does.

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

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the bindings are built as compound bodies)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 must hold for 1 s; it already needs both bindings still and both grippers' finger zones (the fingers, out to
# the pads' outer corners, wherever they are turned) 30 mm or more from them, so the trial cannot end while a gripper
# still holds or touches a binding.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Set this rental snowboard's bindings in a duck stance. Move each binding from its end of the board and set it "
    "down flat over a grey insert grid, highback on the board edge nearest you and disc centred on the black centre "
    "line, so the two discs are 40 cm apart and centred on the red cross. Turn each binding so its toe end points 15 "
    "degrees off straight across the board: the left binding toward the left end, the right one toward the right "
    "end. Then let go of both and lift clear."
)

# -- the board, the cross and the insert dots (fixed) ------------------------------------------------------------
BOARD = [
    Box(name, pos=(x, 0.0, L.TOP + L.BOARD_T / 2), size=(w, L.BOARD_LEN, L.BOARD_T), color=L.COLORS[colour])
    for name, x, w, colour in L.BOARD_STRIPS
]
CROSS = [  # the strip along y sits on the one along x, so it shows over the black centre line
    Box("cross_x", pos=(L.CROSS_X, L.CROSS_Y, L.BOARD_TOP + L.CROSS_T / 2), size=(L.CROSS_LEN, L.CROSS_W, L.CROSS_T),
        color=L.RED),
    Box("cross_y", pos=(L.CROSS_X, L.CROSS_Y, L.BOARD_TOP + 1.5 * L.CROSS_T), size=(L.CROSS_W, L.CROSS_LEN, L.CROSS_T),
        color=L.RED),
]
DOTS = [
    Cylinder(f"insert_{grid}_{i + 1}", pos=(x, y, L.BOARD_TOP + L.DOT_H / 2), radius=L.DOT_R, height=L.DOT_H,
             color=L.DOT_GREY)
    for grid, ys in (("nose", L.DOT_YS_NOSE), ("tail", L.DOT_YS_TAIL))
    for i, (x, y) in enumerate((x, y) for y in ys for x in L.DOT_XS)
]

# -- the bindings (loose compounds) ------------------------------------------------------------------------------
SHAPES = (
    mk.box(L.BASEPLATE[0], L.BASEPLATE[1], color=L.BLACK),
    mk.cylinder(L.DISC_CENTER, L.DISC_R, L.DISC_H, color=L.DISC_GREY),
    mk.box(L.TICK[0], L.TICK[1], color=L.TICK_WHITE),
    mk.box(L.HIGHBACK[0], L.HIGHBACK[1], color=L.BLACK),
    mk.box(L.STRAP[0], L.STRAP[1], color=L.STRAP_GREY),
)
_COLUMN = np.asarray(L.COLUMN_CENTER, dtype=float)


class Binding(mk.Assembly):
    """A binding exactly as mechkit builds and reads it in the simulator. Only tools/offline/check_task.py calls the
    two methods below: offline it is the 'strap column' (the strap bar's 30 x 130 mm plan footprint from the
    underside to the strap's top), the part a gripper closes on, so the checker grasps a 30 mm column, carries it
    turned with the gripper and drops it onto the board; ``offline_state`` rebuilds the binding's frame from that
    column for the hooks."""

    def offline_box(self):
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ _COLUMN
        return tuple(float(v) for v in centre), tuple(float(v) for v in L.COLUMN_SIZE), float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ _COLUMN
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


def _binding(name: str, start) -> Binding:
    x, y, angle = start
    b = Binding(name=name, pos=(x, y, L.BOARD_TOP), color=L.BLACK, mass=L.BINDING_MASS, yaw=math.radians(angle),
                shapes=SHAPES)
    b._check()
    return b


BINDING_NOSE = _binding("binding_nose", L.NOSE_START)
BINDING_TAIL = _binding("binding_tail", L.TAIL_START)
OBJECTS = [*BOARD, *CROSS, *DOTS, BINDING_NOSE, BINDING_TAIL]

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
    """Both bindings set (flat, disc over an insert grid and within 10 mm of the centre line, angle within 3 deg of
    +15 for the one nearer the nose and -15 for the other, still over 0.5 s, both grippers' finger zones 30 mm or more
    away), discs 390 to 410 mm apart and their midpoint within 10 mm of the cross; held 1 s. This runs first every
    control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    nose, tail = objects["binding_nose"], objects["binding_tail"]
    arms = rigstate.arms()
    result = _TRACKER.update(nose, tail, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                       right_tips=arms["right"].tips,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """Each binding is set when it is flat, its disc over an insert grid and centred on the line within 10 mm, at its
    angle within 3 deg, still and let go. 1.0 both set, discs 390-410 mm apart, midpoint within 10 mm of the cross;
    0.75 both set, 380-420 mm, within 20 mm; 0.5 both set otherwise; 0.25 exactly one set; 0 neither."""
    return _TRACKER.result(objects["binding_nose"], objects["binding_tail"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Both arms at once, mirrored. Each grips its own binding's ankle-strap bar across its 30 mm (heel to toe), lifts it,
# carries it inward while turning it 75 deg, sets it down 2 mm above the board, lets go and rises to 17 cm.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN70 = -0.035  # 70 mm apart: 20 mm from each face of the bar, 47 mm from the highback
GRIP = -(L.STRAP[1][0] - 0.008) / 2  # -0.011: a 22 mm command on the 30 mm bar
GRIP_TIP = 0.040  # pad tips this far above the table when gripping: 12 mm above the disc, 25 mm up the bar's sides
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_H = GRIP_TIP + PAD_TIPS  # TCP 0.053 above the table: the binding's underside is 43 mm below the TCP
HELD_DROP = GRIP_H - L.BOARD_T
HOVER_H = 0.150
ENTRY_H = 0.083  # pads 20 mm clear of the bar's faces from here down
CARRY_H = 0.120  # the carried binding's underside 67 mm above the board
PLACE_ENTRY_H = 0.085
PLACE_H = L.BOARD_T + 0.002 + HELD_DROP  # 0.055: let go with the underside 2 mm above the board
# At the end the finger tips are 92 mm above the strap bar: the finger zone (45.1 mm round the gripper's axis at a
# 70 mm opening) is then 46.9 mm from the bar, 30 mm needed.
CLEAR_H = 0.170
# Demo knobs (the demo only, never the scoring): per-arm place trims (dx m, dy m, dyaw deg) to cancel the pads'
# offset from the commanded point (about 8 mm along the closing axis, which at the place is 15 deg off x: about
# 7.7 mm in x) and the arm's trail under the 0.25 kg load (3 to 7 mm). Together they can approach the 10 mm centring
# tolerance, so expect one trim pass: run the demo alone first, read x_off_mm, origin and angle_err_deg in
# TASK_TRACE at the end and set these (dx = -x_off, dy = minus the y error, dyaw = -angle_err).
PLACE_TRIM = {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)}
SIDES = {"left": ("binding_nose", L.NOSE_START, L.FRONT_TARGET_DEG, 1.0),
         "right": ("binding_tail", L.TAIL_START, L.BACK_TARGET_DEG, -1.0)}


def place_point(side: str) -> tuple[float, float, float]:
    """(TCP x, TCP y, gripper yaw) at the place: the disc target plus the strap bar's offset turned to the target
    angle (none: the bar is over the disc), and the gripper turned with the binding from yaw 0 at the pick."""
    _, (_, _, start), target, sign = SIDES[side]
    dx, dy, dyaw = PLACE_TRIM[side]
    a = math.radians(target)
    x = L.LINE_X + math.cos(a) * L.STRAP_U + dx
    y = sign * L.STANCE_W / 2 + math.sin(a) * L.STRAP_U + dy
    return x, y, math.radians(target - start + dyaw)


def _arm(side: str) -> list[Waypoint]:
    """One arm's plan (21.4 s); both arms run the same timeline, mirrored."""
    name, (sx, sy, start), target, sign = SIDES[side]
    a0 = math.radians(start)
    px, py = sx + math.cos(a0) * L.STRAP_U, sy + math.sin(a0) * L.STRAP_U  # the strap's centre at the start
    qx, qy, yaw = place_point(side)

    def at(x, y, h):
        return (x, y, L.TOP + h)

    return [
        Waypoint(2.5, at(px, py, HOVER_H), OPEN70, f"over the {name} strap bar", 0.0),
        Waypoint(1.0, at(px, py, ENTRY_H), OPEN70, "down over the bar", 0.0),
        Waypoint(1.5, at(px, py, GRIP_H), OPEN70, "slowly down the last 3 cm", 0.0),
        Waypoint(0.6, at(px, py, GRIP_H), OPEN70, "settle", 0.0),
        Waypoint(1.5, at(px, py, GRIP_H), GRIP, "grip the strap bar", 0.0),
        Waypoint(0.5, at(px, py, GRIP_H), GRIP, "settle the grip", 0.0),
        Waypoint(2.0, at(px, py, CARRY_H), GRIP, "lift", 0.0),
        Waypoint(4.0, at(qx, qy, CARRY_H), GRIP, "carry it over its grid, turning it to its angle", yaw),
        Waypoint(1.0, at(qx, qy, PLACE_ENTRY_H), GRIP, "down to 3 cm above", yaw),
        Waypoint(2.0, at(qx, qy, PLACE_H), GRIP, "slowly down to 2 mm above the board", yaw),
        Waypoint(0.8, at(qx, qy, PLACE_H), GRIP, "settle", yaw),
        Waypoint(1.5, at(qx, qy, PLACE_H), OPEN70, "let go", yaw),
        Waypoint(0.5, at(qx, qy, PLACE_H), OPEN70, "settle", yaw),
        Waypoint(2.0, at(qx, qy, CLEAR_H), OPEN70, "straight up, clear", yaw),
    ]


def demo():
    """Both arms at once. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
