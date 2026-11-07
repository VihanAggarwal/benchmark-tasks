"""On the station table: a bench-scale length of jointed track. Four fixed sleepers (260 x 25 x 18 mm, along x)
carry two rails 20 mm wide and 25 mm tall: the far rail (fixed, set correctly with 10 mm joints) and the near rail,
whose two fixed lengths end 330 mm apart with a loose 310 mm middle length (aluminium box section, 0.25 kg) between
them. The middle length sits between rail seat shoulders (5 mm of play each side) on the two middle sleepers and
has crept 10 mm to the left: its left joint is shut (0 mm), its right joint 20 mm. A yellow 10 mm steel shim (60 x
10 x 45 mm, 0.21 kg) stands on its edge on the table in front of the track. The job: stand the shim in the right
joint against the fixed rail end and hold it, slide the middle length right until it bears on the shim, let go of
the rail, lift the shim out and put it back on the table where it started. Both joints are then 10 mm.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

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
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Both gaps in band, the shim put away, both grippers' tip points 60 mm or more from both parts, both parts still,
# and the joint set on the gauge earlier in the trial: held 1 s.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "The middle length of the near rail has crept: its left joint is shut and its right joint is wide open. "
    "Set both joints to the 10 mm expansion gap. Stand the yellow 10 mm shim upright in the right joint against "
    "the fixed rail end and hold it there, slide the middle length to the right until it bears on the shim, let go "
    "of the rail, then lift the shim out, put it back on the table where it started, in front of the sleeper "
    "ends, and move both grippers away."
)


def _box(name, x, y, z, size, color, mass=None) -> Box:
    return Box(name, pos=(x, y, L.TOP + z), size=size, color=color, mass=mass)


SLEEPERS = [
    _box(f"sleeper_{tag}", L.SLEEPER_X, y, L.SLEEPER_TOP / 2, L.SLEEPER_SIZE, L.SLEEPER_BROWN)
    for tag, y in zip(("left_end", "left_mid", "right_mid", "right_end"), L.SLEEPER_YS)
]
FIXED_RAILS = [
    _box(f"rail_{side}_fixed", L.RAIL_X, sign * L.FIXED_Y, L.RAIL_Z, (L.RAIL_W, L.FIXED_LEN, L.RAIL_H), L.RAIL_GREY)
    for side, sign in (("left", 1.0), ("right", -1.0))
]
MIDDLE = _box("middle_rail", L.MID_X0, L.MID_Y0, L.RAIL_Z, L.MID_SIZE, L.RAIL_GREY, L.MID_MASS)
SHOULDERS = [
    _box(f"shoulder_{side}_{edge}", x, y, L.SHOULDER_Z, L.SHOULDER_SIZE, L.SHOULDER_GREY)
    for side, y in (("left", L.SEAT_YS[0]), ("right", L.SEAT_YS[1]))
    for edge, x in (("near", L.SHOULDER_XS[0]), ("far", L.SHOULDER_XS[1]))
]
FAR_RAIL = [
    _box("far_rail_left", L.FAR_RAIL_X, L.FIXED_Y, L.RAIL_Z, (L.RAIL_W, L.FIXED_LEN, L.RAIL_H), L.RAIL_GREY),
    _box("far_rail_middle", L.FAR_RAIL_X, 0.0, L.RAIL_Z, (L.RAIL_W, L.FAR_MID_LEN, L.RAIL_H), L.RAIL_GREY),
    _box("far_rail_right", L.FAR_RAIL_X, -L.FIXED_Y, L.RAIL_Z, (L.RAIL_W, L.FIXED_LEN, L.RAIL_H), L.RAIL_GREY),
]
SHIM = _box("shim", L.SHIM_X0, L.SHIM_Y0, L.SHIM_SIZE[2] / 2, L.SHIM_SIZE, L.SHIM_YELLOW, L.SHIM_MASS)
OBJECTS = [*SLEEPERS, *FIXED_RAILS, MIDDLE, *SHOULDERS, *FAR_RAIL, SHIM]

_TRACKER = Tracker()


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
    """Middle length seated, shim put away, both gaps 8.5 to 11.5 mm, both grippers' pad-tip points 60 mm or more
    from both parts, both parts still (path under 2 mm over 0.5 s), and the joint set on the gauge at some earlier
    step ('gauged'); held 1 s. This runs first every control step, so it also feeds the tracker (with the arms
    rigstate read for this step)."""
    rail, shim = objects["middle_rail"], objects["shim"]
    arms = rigstate.arms()
    result = _TRACKER.update(rail, shim, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for key, s in (("rail", rail), ("shim", shim)):
        for attr in ("pos", "quat_xyzw", "lin_vel"):
            row[f"{key}_{attr}"] = getattr(s, attr, None)
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the middle length is not seated or a gap is under 3 mm; 0.25 if the shim still stands in the right
    joint; 0 if it is anywhere else on the track. Otherwise 1.0 both gaps 8.5 to 11.5 mm, shim put away, hands off
    (tip points 60 mm clear), both parts still and gauged during the trial; 0.5 both gaps 6 to 14 mm; else 0.25."""
    return _TRACKER.result(objects["middle_rail"], objects["shim"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
SHIM_GRIP = -(L.SHIM_SIZE[0] - 0.008) / 2  # across the shim's 60 mm: -0.026, 20 N per finger
RAIL_GRIP = -(L.RAIL_W - 0.008) / 2  # across the rail's 20 mm: -0.006
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # both grippers close their fingers along x (across the track)
TRAVEL_Z = L.TOP + 0.10
PARK_Z = L.TOP + 0.14

# Right arm: the shim. Gripped with the TCP 30 mm up (pad tips 17 mm up, the pads on its 10 x 45 mm ends, 7.5 mm
# above its centre), it hangs 30 mm below the TCP.
SHIM_HANG = 0.030  # TCP height above the shim's underside while it is held
SHIM_PICK_Z = L.TOP + SHIM_HANG  # its underside on the table
SHIM_ENTRY_Z = L.TOP + L.SLEEPER_TOP + L.RAIL_H + 0.002 + SHIM_HANG  # 0.075: its underside 2 mm above the rail tops
SHIM_SET_Z = L.TOP + 0.0015 + SHIM_HANG  # 0.0315: in the joint, its underside 1.5 mm above the table
SHIM_PUT_Z = L.TOP + 0.0005 + SHIM_HANG  # put down with its underside 0.5 mm above the table
JOINT_Y = ((L.MID_Y0 - L.HALF_LEN) + (-L.FIX_END_Y)) / 2  # -0.155: the middle of the 20 mm open joint
SEAT_Y = -L.FIX_END_Y + L.SHIM_THICK / 2  # -0.160: the shim's face on the fixed rail end's face
PARK_R = (0.150, -0.250)

# Left arm: the middle length. Gripped across its 20 mm width at y +0.135 (30 mm in from its left end, outboard of
# the left middle sleeper and its shoulders), TCP 40 mm up (pad tips 27 mm up: 16 mm of pad on the rail). Nearer the
# track's centre the two arms' forearms would meet (capsule clearance 11 mm at y +0.045, 77 mm at +0.104).
GRIP_Y = 0.135
GRIP_Z = L.TOP + 0.040
GRIP_ENTRY_Z = L.TOP + 0.060
TRAVEL_TO_SHIM = L.START_RIGHT_GAP - L.SHIM_THICK  # 0.010: the slide that brings the rail onto the shim
# The arm trails a push: the compliance model (J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad) gives about 5 mm per
# newton along y at this grip, and pushing toward -y also presses the gripper down on the rail (about 0.5 N down per
# N along y with the rail held), which adds friction. So the slide is commanded far past the shim (SLIDE_CMD, 27 mm:
# the rail reaches the shim for any trail up to 17 mm, and its push on the shim, backed by the fixed rail end, is
# then about 3-6 N), with the grip raised LIFT_BIAS (2 mm) while it pushes so that the gripper does not press the rail
# into its seats (gripped outboard of the left sleeper it can take about 1 N of the rail's 2.45 N weight before the
# rail tips about the right sleeper, which a 2 mm rise allows by about 1 mm at most). Then the push is eased off,
# back to where the rail stopped (10 mm), so that it is let go with no load: nothing drags it back or pushes it on.
# Watch rail_y_mm in TASK_TRACE: it must reach 0 and stay there through the release; 'gauged' must turn true
# while the rail is held on the shim (shim_to_fixed_end_mm and shim_to_rail_mm both 1 mm or less).
SLIDE_CMD = 0.027
LIFT_BIAS = 0.002
EASE_TO = TRAVEL_TO_SHIM
PARK_L = (0.200, 0.220)


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _right_seat() -> list[Waypoint]:
    """Right arm, first part: pick the shim at its 60 mm ends, carry it over the open right joint, lower it slowly
    into the joint (5 mm clear each side) to just above the table, then move it 5 mm right onto the fixed rail
    end's face."""
    sx, sy = L.SHIM_X0, L.SHIM_Y0
    x = L.RAIL_X
    a = YAW_R
    return [
        Waypoint(2.5, (sx, sy, TRAVEL_Z), OPEN, "over the shim", a),
        Waypoint(2.0, (sx, sy, L.TOP + 0.050), OPEN, "down toward it", a),
        Waypoint(1.5, (sx, sy, SHIM_PICK_Z), OPEN, "slowly down round its ends", a),
        Waypoint(0.6, (sx, sy, SHIM_PICK_Z), OPEN, "settle", a),
        Waypoint(1.5, (sx, sy, SHIM_PICK_Z), SHIM_GRIP, "grip the shim at its ends", a),
        Waypoint(0.5, (sx, sy, SHIM_PICK_Z), SHIM_GRIP, "settle", a),
        Waypoint(2.0, (sx, sy, TRAVEL_Z), SHIM_GRIP, "lift it", a),
        Waypoint(2.5, (x, JOINT_Y, TRAVEL_Z), SHIM_GRIP, "carry it over the open right joint", a),
        Waypoint(2.0, (x, JOINT_Y, SHIM_ENTRY_Z), SHIM_GRIP, "down to just above the rail tops", a),
        Waypoint(0.5, (x, JOINT_Y, SHIM_ENTRY_Z), SHIM_GRIP, "settle", a),
        Waypoint(2.0, (x, JOINT_Y, SHIM_SET_Z), SHIM_GRIP, "slowly down into the joint", a),
        Waypoint(0.6, (x, JOINT_Y, SHIM_SET_Z), SHIM_GRIP, "settle", a),
        Waypoint(1.5, (x, SEAT_Y, SHIM_SET_Z), SHIM_GRIP, "onto the fixed rail end's face", a),
    ]


RIGHT_SEATED_S = _span(_right_seat())  # 19.7 s: the shim stands in the joint against the fixed rail end


def _left() -> list[Waypoint]:
    """Left arm: grip the middle length across its width once the shim is nearly in place, then (0.5 s after the
    shim is on the fixed rail end) slide the rail right onto the shim, ease the push off, let go, go straight up and
    park."""
    x, y = L.RAIL_X, GRIP_Y
    a = YAW_L
    approach = [
        Waypoint(2.5, (x, y, TRAVEL_Z), OPEN, "over the middle length", a),
    ]
    descend = [
        Waypoint(2.0, (x, y, GRIP_ENTRY_Z), OPEN, "down toward it", a),
        Waypoint(1.5, (x, y, GRIP_Z), OPEN, "slowly down round it", a),
        Waypoint(0.6, (x, y, GRIP_Z), OPEN, "settle", a),
        Waypoint(1.5, (x, y, GRIP_Z), RAIL_GRIP, "grip the rail across its width", a),
        Waypoint(0.5, (x, y, GRIP_Z), RAIL_GRIP, "settle", a),
    ]
    wait = RIGHT_SEATED_S + 0.5 - _span(approach) - _span(descend)
    ease_y = y - EASE_TO
    push = (x, y - SLIDE_CMD, GRIP_Z + LIFT_BIAS)  # commanded 17 mm past the shim, the grip 2 mm higher
    return approach + [
        Waypoint(wait, (x, y, TRAVEL_Z), OPEN, "wait until the shim is nearly in place", a),
    ] + descend + [
        Waypoint(4.0, push, RAIL_GRIP, "slide it right until it bears on the shim", a),
        Waypoint(1.0, push, RAIL_GRIP, "hold it on the shim", a),
        Waypoint(2.0, (x, ease_y, GRIP_Z), RAIL_GRIP, "ease the push off, back to where it stopped", a),
        Waypoint(0.5, (x, ease_y, GRIP_Z), RAIL_GRIP, "settle", a),
        Waypoint(1.5, (x, ease_y, GRIP_Z), OPEN, "let go of the rail", a),
        Waypoint(0.5, (x, ease_y, GRIP_Z), OPEN, "settle", a),
        Waypoint(1.5, (x, ease_y, TRAVEL_Z), OPEN, "straight up, clear of the rail", a),
        Waypoint(2.0, (*PARK_L, PARK_Z), OPEN, "park out of the way", a),
    ]


def _left_clear_s() -> float:
    """When the left gripper has let go of the rail and risen straight up clear of it."""
    t = 0.0
    for w in _left():
        t += w.duration
        if w.label == "straight up, clear of the rail":
            return t
    raise AssertionError("no lift in the left plan")


LEFT_CLEAR_S = _left_clear_s()  # 31.2 s


def _right() -> list[Waypoint]:
    """Right arm: seat the shim, hold it there until the left gripper has let go of the rail and risen clear, then
    lift the shim straight out, carry it back in front of the track, put it down, let go and park."""
    sx, sy = L.SHIM_X0, L.SHIM_Y0
    x = L.RAIL_X
    a = YAW_R
    seat = _right_seat()
    return seat + [
        Waypoint(LEFT_CLEAR_S - RIGHT_SEATED_S, (x, SEAT_Y, SHIM_SET_Z), SHIM_GRIP,
                 "hold it while the rail is slid onto it and let go", a),
        Waypoint(2.0, (x, SEAT_Y, SHIM_ENTRY_Z), SHIM_GRIP, "slowly lift it out of the joint", a),
        Waypoint(1.0, (x, SEAT_Y, TRAVEL_Z), SHIM_GRIP, "lift it clear", a),
        Waypoint(2.5, (sx, sy, TRAVEL_Z), SHIM_GRIP, "carry it back in front of the track", a),
        Waypoint(2.0, (sx, sy, L.TOP + 0.050), SHIM_GRIP, "down toward the table", a),
        Waypoint(1.5, (sx, sy, SHIM_PUT_Z), SHIM_GRIP, "slowly down onto the table", a),
        Waypoint(0.6, (sx, sy, SHIM_PUT_Z), SHIM_GRIP, "settle", a),
        Waypoint(1.5, (sx, sy, SHIM_PUT_Z), OPEN, "let go of the shim", a),
        Waypoint(0.5, (sx, sy, SHIM_PUT_Z), OPEN, "settle", a),
        Waypoint(1.5, (sx, sy, TRAVEL_Z), OPEN, "straight up", a),
        Waypoint(2.0, (*PARK_R, PARK_Z), OPEN, "park out of the way", a),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
