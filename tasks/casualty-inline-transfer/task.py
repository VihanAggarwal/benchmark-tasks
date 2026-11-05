"""On the station table: a grey floor mat (120 x 380 x 3 mm, fixed) 20 cm in front of the arm bases, just right of
the midline, with a simplified casualty manikin lying on its back on it along y, head toward the robot's left: a
rigid trunk (70 x 240 x 40 mm, 0.22 kg) and a rigid head (60 x 70 x 44 mm, with a short neck bar; 0.07 kg) joined
by one free pin hinge across the body (the neck: +-40 deg, a light return spring). Beyond it, 31 to 42 cm out, a
fixed blue stretcher (110 x 400 x 30 mm) with a white head zone near its left end. In-line rule: the manikin goes
onto the stretcher with the head on the head zone, the neck kept straight the whole way (the score is the worst
bend), the head and the trunk lifted and lowered together, their undersides never more than 8 cm above the table,
and never held by the neck; at the end it lies on its back.

The manikin is a mechkit assembly (mechkit.py, next to this file: a floating articulation with one revolute
joint); the scorer reads its joint and link poses from it, and the arms (tool point, finger tips, opening) through
rigstate.py, which wraps the embodiment's step and reset from this file's import. Neither changes what the
embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the manikin is built as an articulation)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with both grippers let go (open, finger tips 30 mm or more from the manikin): the trial cannot end
# while a gripper still holds or touches the manikin.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Move the manikin from the grey floor mat onto the blue stretcher, on its back with its head on the white head "
    "zone. Its neck is hinged: keep the head in line with the body the whole time (neck bent no more than 6 degrees), "
    "lift and lower the head and the body together, never raise its underside more than 8 cm above the table, and "
    "never hold it by the neck. Then let go of it with both grippers."
)

MAT = Box("floor_mat", pos=(L.MAT_X, L.MAT_Y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE, color=L.MAT_GREY)

# The manikin: the trunk is the base body; the head link (neck bar + head) turns about the trunk's x axis through
# the hinge at the head end of the trunk, mid-height. Positive neck angles lift the head's crown.
NECK = mk.hinge(
    "neck", "X", L.NECK_ANCHOR,
    [mk.box(L.NECK_BAR_CENTER, L.NECK_BAR_SIZE, color=L.NECK_BEIGE), mk.box(L.HEAD_CENTER, L.HEAD_SIZE)],
    lower=-L.NECK_LIMIT, upper=L.NECK_LIMIT, mass=L.HEAD_MASS, spring=L.NECK_SPRING, damping=L.NECK_DAMPING,
    color=L.SKIN,
)
MANIKIN = mk.assembly("manikin", origin=L.MANIKIN_ORIGIN, shapes=[mk.box(L.TRUNK_CENTER, L.TRUNK_SIZE)],
                      mass=L.TRUNK_MASS, joints=[NECK], color=L.SKIN)


def _stretcher_part(name: str, y0: float, y1: float, color) -> Box:
    return Box(name, pos=(L.STRETCHER_X, (y0 + y1) / 2, L.TOP + L.STRETCHER_TOP / 2),
               size=(L.STRETCHER_WIDTH, y1 - y0, L.STRETCHER_TOP), color=color)


STRETCHER = [  # one flush 30 mm slab in three pieces: bed, white head zone, top end
    _stretcher_part("stretcher_bed", L.STRETCHER_Y0, L.HEAD_ZONE_Y0, L.STRETCHER_BLUE),
    _stretcher_part("stretcher_head_zone", L.HEAD_ZONE_Y0, L.HEAD_ZONE_Y1, L.HEAD_ZONE_WHITE),
    _stretcher_part("stretcher_top_end", L.HEAD_ZONE_Y1, L.STRETCHER_Y1, L.STRETCHER_BLUE),
]
OBJECTS = [MAT, MANIKIN, *STRETCHER]

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
    """Manikin on the stretcher, head on the head zone, the neck never bent more than 6 deg and within 3 deg now,
    both grippers let go, no rule broken at any time; held still 1 s. This runs first every control step, so it
    also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    m = objects["manikin"]
    arms = rigstate.arms()
    result = _TRACKER.update(m, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(neck_rad=float(m.joints["neck"]), neck_vel=float(m.joint_vel["neck"]), origin=m.origin,
                   head_link=m.links.get("neck"), trunk_vel=m.lin_vel)
    except (AttributeError, KeyError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips,
                   grippers={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, the neck bent more than 20 deg, a body went more than 80 mm above the table, the head or
    the trunk was lifted more than 15 mm while the other still rested, a gripper was at the neck while the manikin
    was lifted, a body fell, or the manikin left the table. Else, on the stretcher: 1.0 neck never over 6 deg, within
    3 deg now and both grippers let go; 0.75 never over 10 deg, within 5 deg now; 0.5 never over 15 deg; 0.25
    otherwise. Not on it: 0.25 if both bodies were once up 20 mm together with the neck within 10 deg; else 0."""
    return _TRACKER.result(objects["manikin"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
HEAD_GRIP = -(L.HEAD_SIZE[0] - 0.008) / 2  # across the head's 60 mm: 52 mm
TRUNK_GRIP = -(L.TRUNK_SIZE[0] - 0.008) / 2  # across the trunk's 70 mm: 62 mm
# Both grips close the fingers along x (yaw -pi/2 left, +pi/2 right, the signs that reach toward the midline), so
# the open fingers stand in front of and behind the manikin, never toward the other arm or over the neck.
YAW_L = -math.pi / 2
YAW_R = math.pi / 2
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
# The two grips are 0.230 m apart at one x: the head 15 mm from its centre toward the crown (the fingers' 27 mm width
# ends 6.5 mm short of the crown), the trunk 30 mm from its centre toward the hips. At 0.205 m apart (the head's and
# the trunk's centres plus 20 mm) the two wrists' (link4) collision capsules come within 6 mm of each other at x
# 0.20, at 0.220 m within 23 mm; at 0.230 m they stay 34 mm apart for the 8 s the arms spend at x 0.20. The grips
# are 65 mm and 165 mm from the neck bar's centre, and the off-centre grips turn each body by well under a degree
# (0.065 N m on the trunk, 0.013 N m on the head, against about 9 N m/rad of arm stiffness about x at the TCP).
HEAD_XY = (L.MANIKIN_ORIGIN[0], L.MANIKIN_ORIGIN[1] + L.HEAD_CENTER[1] + 0.015)  # (0.200, +0.115)
TRUNK_XY = (L.MANIKIN_ORIGIN[0], L.MANIKIN_ORIGIN[1] - 0.030)  # (0.200, -0.115)
GRIP_Z = L.TOP + 0.025  # pad tips 12 mm above the table (9 mm over the mat), pads 12-34 mm up both bodies' sides
HELD_DROP = GRIP_Z - (L.TOP + L.MAT_TOP)  # a held body's underside is 22 mm below the TCP
ENTRY_Z = L.TOP + 0.045  # the last 20 mm of the descent are a separate 1.5 s move
HOVER_Z = L.TOP + 0.100
LIFT = 0.050  # both bodies 50 mm up: undersides 23 mm over the stretcher's top, 27 mm under the 80 mm cap
# Payload sag: the arms' controllers do not carry a payload's weight, and the right arm holds the heavier body
# (0.22 kg against 0.07 kg), so it is commanded 2 mm higher once loaded to keep the two bodies level (the neck
# straight). On the GPU read lift_trunk_mm / lift_head_mm and neck_deg in TASK_TRACE during the carry and change
# SAG_TRIM_R so the two lifts match.
SAG_TRIM_R = 0.002
CARRY_Z = GRIP_Z + LIFT
PLACE_Z = L.TOP + L.STRETCHER_TOP + 0.001 + HELD_DROP  # set down with both undersides 1 mm over the stretcher
PLACE_DX = L.STRETCHER_X - L.MANIKIN_ORIGIN[0]  # 0.165 m straight out along x: the head ends on the head zone
CLEAR_Z = L.TOP + 0.140  # finger tips 53 mm over the head's top (57 mm over the trunk's) at the end


def _arm(xy, grip, yaw, trim, what) -> list[Waypoint]:
    """One arm's plan: both arms run the same timeline (21 s), so the two bodies go up, across and down together."""
    x, y = xy
    px = x + PLACE_DX
    return [
        Waypoint(3.0, (x, y, HOVER_Z), OPEN, f"over the {what}", yaw),
        Waypoint(2.0, (x, y, ENTRY_Z), OPEN, f"down beside the {what}", yaw),
        Waypoint(1.5, (x, y, GRIP_Z), OPEN, "slowly down the last 2 cm", yaw),
        Waypoint(0.5, (x, y, GRIP_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, GRIP_Z), grip, f"grip the {what}", yaw),
        Waypoint(0.5, (x, y, GRIP_Z), grip, "settle the grip", yaw),
        Waypoint(2.0, (x, y, CARRY_Z + trim), grip, "lift 5 cm, together with the other arm", yaw),
        Waypoint(3.0, (px, y, CARRY_Z + trim), grip, "carry it over the stretcher, level", yaw),
        Waypoint(2.0, (px, y, PLACE_Z + trim), grip, "lower onto the stretcher, together", yaw),
        Waypoint(0.5, (px, y, PLACE_Z + trim), grip, "settle", yaw),
        Waypoint(1.5, (px, y, PLACE_Z + trim), OPEN, "let go", yaw),
        Waypoint(1.0, (px, y, PLACE_Z + trim), OPEN, "settle", yaw),
        Waypoint(2.0, (px, y, CLEAR_Z), OPEN, "straight up, clear", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: the head, held across its 60 mm, 15 mm from its centre toward the crown."""
    return _arm(HEAD_XY, HEAD_GRIP, YAW_L, 0.0, "head")


def _right() -> list[Waypoint]:
    """Right arm: the trunk, held across its 70 mm, 30 mm from its centre toward the hips."""
    return _arm(TRUNK_XY, TRUNK_GRIP, YAW_R, SAG_TRIM_R, "trunk")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
