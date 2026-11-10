"""On the station table: a 2:1 bench mock-up of a pendulum impact (Charpy) tester's specimen area on the midline,
about 30 cm in front of the arm bases: a dark cast-iron base carrying two steel supports and, behind them, two steel
anvils with an 80 mm gap between them; a red strip on the base in front of the gap marks the striker's side. On a
black mat on the right lies the specimen: an aluminium 2:1 Charpy bar (110 x 20 x 20 mm) with a 4 x 4 mm notch
across one long side at mid-length, lying along x with that notch toward -y. Rule: lay it on both supports along
the line between the anvils, notch facing away from the striker (toward the anvils), the notch centred between the
anvils and the bar against both anvils; then let go of it and move clear.

The specimen is a loose mechkit compound (mechkit.py, next to this file): one rigid body of three boxes and a thin
black tick, so its notch is a real slot and the scorer can read its pose (base frame and orientation). The scorer
also reads the arms (TCP, pad-tip point, opening) through rigstate.py, which wraps the embodiment's step and reset
from this file's import; neither changes anything else the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the task needs only one; the demo uses the right arm and the left arm holds its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the specimen is built as a compound body)
import rigstate  # noqa: E402
from scoring import Tracker, ends, notch_normal, notch_offset  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s, with both grippers' finger zones at least 20 mm from the specimen (part of every tier), so the trial
# cannot end while a gripper still holds or touches it.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Load the Charpy specimen into the impact tester: lay it across the two supports, square against both anvils, "
    "with its notch facing away from the striker (its side is marked by the red strip) and centred between the "
    "anvils. Then let go of it and move clear of it."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color, **kw)


BASE = _span_box("tester_base", L.BASE_X0, L.BASE_X1, -L.BASE_HALF_Y, L.BASE_HALF_Y, 0.0, L.BASE_H, L.CAST_IRON)
SUPPORTS = [
    _span_box(f"support_{side}", L.SUP_X0, L.SUP_X1, *sorted((s * L.SUP_Y_IN, s * L.SUP_Y_OUT)), L.BASE_H, L.SUP_TOP,
              L.HARDENED_STEEL)
    for side, s in (("left", 1.0), ("right", -1.0))
]
ANVILS = [
    _span_box(f"anvil_{side}", L.FACE_X, L.ANVIL_X1, *sorted((s * L.ANVIL_Y_IN, s * L.ANVIL_Y_OUT)), L.BASE_H,
              L.ANVIL_TOP, L.HARDENED_STEEL)
    for side, s in (("left", 1.0), ("right", -1.0))
]
STRIKER_MARK = Box("striker_mark", pos=(L.STRIKER_MARK_CENTER[0], L.STRIKER_MARK_CENTER[1],
                                        L.TOP + L.BASE_H + L.STRIKER_MARK_SIZE[2] / 2),
                   size=L.STRIKER_MARK_SIZE, color=L.STRIKER_RED)
MAT = Box("specimen_mat", pos=(L.MAT_X, L.MAT_Y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE, color=L.MAT_BLACK)
SPECIMEN = mk.assembly(
    "specimen",
    origin=(L.SP_START[0], L.SP_START[1], L.TOP + L.SP_START[2]),
    yaw=L.SP_START_YAW,
    mass=L.SP_MASS,
    color=L.ALUMINIUM,
    shapes=[
        mk.box(L.BACK_CENTER, L.BACK_SIZE),
        mk.box(L.SIDE_CENTERS[0], L.SIDE_SIZE),
        mk.box(L.SIDE_CENTERS[1], L.SIDE_SIZE),
        mk.box(L.TICK_CENTER, L.TICK_SIZE, color=L.TICK_BLACK),
    ],
)
OBJECTS = [BASE, *SUPPORTS, *ANVILS, STRIKER_MARK, MAT, SPECIMEN]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and a teleport back to the start are
    two more signals)."""
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
    """The specimen lies flat on both supports along the line between the anvils, notch away from the striker,
    centred within 2 mm, the notched face within 1.5 mm of both anvils, at rest (neither end moving more than 2 mm in
    0.5 s), both grippers' finger zones 20 mm or more from it; held 1 s. This runs first every control step, so it
    also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    sp = objects["specimen"]
    arms = rigstate.arms()
    result = _TRACKER.update(sp, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    try:
        a, b = ends(sp)
        row.update(source=getattr(sp, "source", None), sp_origin=sp.origin, sp_quat=sp.quat_xyzw, sp_vel=sp.lin_vel,
                   end_a=a, end_b=b, notch_y_mm=round(notch_offset(sp) * 1000, 2),
                   notch_yaw_deg=round(math.degrees(math.atan2(*notch_normal(sp)[1::-1])), 2))
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips, openings=[arms["left"].opening, arms["right"].opening])
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the specimen ends at rest, let go (both grippers' finger zones 20 mm or more from it), lying on both
    supports (notch running vertically within 5 degrees, length within 10 degrees of the line between the anvils,
    each end at least 3 mm onto its support with its underside within 2 mm of the support's top), its notch facing
    away from the striker (the notched face's normal within 30 degrees of +x) and the notch within 8 mm of the
    midplane between the anvils. Then 0.5, +0.25 if the notch is within 2 mm of the midplane, +0.25 if the notched
    face is within 1.5 mm of both anvils' faces. Read from the state at hand (the last step)."""
    return _TRACKER.result(objects["specimen"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
SP_OPEN = -0.025  # 50 mm: 15 mm either side of the 20 mm bar
SP_GRIP = -(L.SP_W - 0.008) / 2  # 12 mm commanded across the bar's 20 mm
# Picked at its middle where it lies along x, the fingers closing along y (yaw 0) on the notched face and the back
# face, and turned a quarter turn to the left (yaw +pi/2) while it is carried: the bar swings from x to y and the
# notched face from -y to +x, toward the anvils. That is the right arm's only workable pair of headings: at the pick
# yaw pi fails (joint limits), and over the anvils yaw -pi/2 runs joint6 into its limit.
YAW_PICK, YAW_PLACE = 0.0, math.pi / 2
# Trim of the place yaw, for the GPU run. The bar lies between flat pads, so it takes the pads' heading, and the
# 1.5 mm feeler over its 110 mm length allows only 0.78 degrees of skew (one end on its anvil, the other end off).
# In our GPU runs, parts carried in a gripper mostly ended turned counter-clockwise from the commanded yaw, by about
# +0.6 to +1.7 degrees (both arms); in a matching right-arm job (picked at yaw 0 near here, laid at yaw +pi/2 near
# the place point) the bar turned +1.32 degrees as the pads closed, was +1.54 degrees in the hand at the place and
# +1.39 at rest. Pressing at the bar's middle cannot square it (about 4 N would be needed against the grip and the
# 10 N m/rad wrist), so the place yaw is set 1.2 degrees clockwise: any bias from +0.45 to +1.95 degrees then leaves
# under 0.78 degrees of skew. Offline (no bias) the bar is laid 1.2 degrees turned. If notch_yaw_deg in TASK_TRACE
# after 'against the anvils' is more than 0.5 degrees from 0, change YAW_TRIM by minus that many degrees.
YAW_TRIM = math.radians(-1.2)
YAW_SET = YAW_PLACE + YAW_TRIM
GRIP_UP = 0.017  # the TCP 17 mm above the bar's underside: pad tips 4 mm above it (and the mat)
APPROACH = 0.030  # each final descent starts this far above its end height
HOVER = L.TOP + 0.12  # carried with the bar's underside 103 mm up: 33 mm over the anvils' tops
PICK = (L.SP_START[0], L.SP_START[1], L.TOP + L.MAT_TOP + GRIP_UP)
# Over the tester the bar comes down beside the anvils with its notched face CLEAR in front of them (so it cannot
# land on an anvil's top), stops with its underside SIDE_UP above the supports, moves toward the anvils until its
# notched face is aimed PRESS past their faces (they stop it: the arm's lateral compliance, about 6 mm/N at this pose,
# turns the excess into a push of about 0.2-0.4 N), is lowered onto the supports while pressed against them (aimed
# DROP above), and let go. The anvils, not the arm's accuracy, set the 1.5 mm gap rule: any arm error from -4 mm
# (toward the striker) upward still ends in contact.
CLEAR = 0.004
SIDE_UP = 0.008
PRESS = 0.0025
DROP = 0.001
# Trim of the place point (x, y), for the GPU run. The compliance model (J K^-1 J^T of the 0.118 kg bar's weight,
# joints 80/80/80/10/10/10 N m/rad, yamkin) puts the gripper (-0.9, -0.9, -3.5) mm off at the place pose. x needs no
# trim (PRESS covers it); y is trimmed by about half the modelled sideways sag so the notch lands within 0.5 mm of
# the midplane either way (0.4 mm offline, no sag). If offset_mm in TASK_TRACE is off by more than 1 mm, move the y
# trim by minus that offset.
PLACE_TRIM = (0.0, 0.0004)
X_CLEAR = L.FACE_X - CLEAR - L.SP_W / 2 + PLACE_TRIM[0]  # 0.316: the bar's centre, its notched face 4 mm off
X_PRESS = L.FACE_X + PRESS - L.SP_W / 2 + PLACE_TRIM[0]  # 0.3225: the notched face aimed 2.5 mm past the anvils
Y_PLACE = L.MID_Y + PLACE_TRIM[1]  # the grip is at the notch: the notch's y is the TCP's y
Z_SIDE = L.TOP + L.SUP_TOP + SIDE_UP + GRIP_UP
Z_SET = L.TOP + L.SUP_TOP + DROP + GRIP_UP
UP = L.TOP + 0.10
PARK = (0.22, -0.20, L.TOP + 0.15)


def _right() -> list[Waypoint]:
    """Right arm: grip the bar at its middle, lift it, carry it over the supports while turning it a quarter turn
    left, lower it beside the anvils, push it against them, lower it onto the supports, let go, go up and aside."""
    px, py, pz = PICK
    return [
        Waypoint(2.0, (px, py, HOVER), SP_OPEN, "over the specimen", YAW_PICK),
        Waypoint(1.0, (px, py, pz + APPROACH), SP_OPEN, "down to just above it", YAW_PICK),
        Waypoint(1.5, (px, py, pz), SP_OPEN, "slowly down beside it", YAW_PICK),
        Waypoint(0.5, (px, py, pz), SP_OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (px, py, pz), SP_GRIP, "grip the specimen", YAW_PICK),
        Waypoint(1.5, (px, py, HOVER), SP_GRIP, "lift it", YAW_PICK),
        Waypoint(2.5, (X_CLEAR, Y_PLACE, HOVER), SP_GRIP, "carry it over the supports, notch turning to the anvils",
                 YAW_SET),
        Waypoint(0.8, (X_CLEAR, Y_PLACE, HOVER), SP_GRIP, "settle over the supports", YAW_SET),
        Waypoint(1.0, (X_CLEAR, Y_PLACE, Z_SIDE + APPROACH), SP_GRIP, "down to 3 cm above", YAW_SET),
        Waypoint(2.0, (X_CLEAR, Y_PLACE, Z_SIDE), SP_GRIP, "slowly down beside the anvils", YAW_SET),
        Waypoint(1.5, (X_PRESS, Y_PLACE, Z_SIDE), SP_GRIP, "against the anvils", YAW_SET),
        Waypoint(0.5, (X_PRESS, Y_PLACE, Z_SIDE), SP_GRIP, "settle", YAW_SET),
        Waypoint(1.5, (X_PRESS, Y_PLACE, Z_SET), SP_GRIP, "slowly onto the supports, pressed to the anvils",
                 YAW_SET),
        Waypoint(0.5, (X_PRESS, Y_PLACE, Z_SET), SP_GRIP, "settle", YAW_SET),
        Waypoint(1.5, (X_PRESS, Y_PLACE, Z_SET), SP_OPEN, "let go of it", YAW_SET),
        Waypoint(1.0, (X_PRESS, Y_PLACE, UP), SP_OPEN, "straight up", YAW_SET),
        Waypoint(1.5, PARK, SP_OPEN, "aside", YAW_SET),
    ]


def demo():
    """One arm: the left arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
