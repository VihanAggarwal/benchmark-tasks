"""On the station table: a three-point bend (flexural) test fixture on the midline, about 31 cm in front of the arm
bases: a dark steel base (90 x 300 x 10 mm) with a red centre line across it, where the testing machine's loading nose
comes down, and a scale of white ticks every 10 mm along its near edge; on it two loose blue support blocks, each with
a bright steel roller along x on top, 64 mm apart (as set for the last specimen, a 4 mm thick bar). On a black mat in
front of the left arm lies the next specimen: a cream glass-filled nylon bar, 210 x 20 x 10 mm, along y. Rule
(ASTM D790 / ISO 178): the support span is 16 times the specimen's depth, so for this 10 mm bar the rollers go
160 mm apart, symmetric about the centre line; the bar is laid flat across both rollers, square to them, its middle
midway between them; then everything is let go.

The support blocks are loose mechkit compounds (mechkit.py, next to this file): one rigid body each, a block and a
lying roller, so the scorer can read where each roller's axis is. The scorer also reads the arms (TCP, pad-tip point,
opening) through rigstate.py, which wraps the embodiment's step and reset from this file's import; neither changes
anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the support blocks are compound bodies)
import rigstate  # noqa: E402
from scoring import Tracker, roller_centre, sp_ends  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s, with both grippers' finger zones at least 20 mm from the specimen and both supports (part of every
# tier), so the trial cannot end while a gripper still holds or touches any of them.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Set the bend fixture for this 10 mm thick specimen: move the two support blocks so their rollers are 16 times "
    "its thickness apart, centred on the red centre line, then lay the specimen flat across the rollers, centred "
    "and square to them. Let go of everything."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color, **kw)


BASE = _span_box("fixture_base", L.BASE_X0, L.BASE_X1, -L.BASE_HALF_Y, L.BASE_HALF_Y, 0.0, L.BASE_H, L.BASE_STEEL)
CENTRE_LINE = _span_box("centre_line", L.BASE_X0, L.BASE_X1, L.MID_Y - L.CENTRE_LINE_W / 2,
                        L.MID_Y + L.CENTRE_LINE_W / 2, L.BASE_H, L.BASE_H + L.MARK_T, L.LINE_RED)


def _tick(i: int, sign: float) -> Box:
    """The scale tick i x 10 mm from the centre line toward +y (sign +1) or -y (sign -1)."""
    length = L.TICK_LEN_LONG if i % L.TICK_LONG_EVERY == 0 else L.TICK_LEN_SHORT
    y = L.MID_Y + sign * i * L.TICK_PITCH
    return _span_box(f"scale_tick_{'p' if sign > 0 else 'm'}{i * 10:03d}", L.TICK_X0, L.TICK_X0 + length,
                     y - L.TICK_W / 2, y + L.TICK_W / 2, L.BASE_H, L.BASE_H + L.MARK_T, L.TICK_WHITE)


TICKS = [_tick(i, s) for s in (1.0, -1.0) for i in range(1, L.TICK_COUNT + 1)]


def _support(side: str) -> mk.Assembly:
    x, y = L.SUP_START[side]
    return mk.assembly(
        f"support_{side}",
        origin=(x, y, L.BASE_TOP_W),
        yaw=L.SUP_START_YAW,
        mass=L.SUP_MASS,
        color=L.BLOCK_BLUE,
        shapes=[
            mk.box(L.BLOCK_CENTER, L.BLOCK_SIZE),
            mk.cylinder((0.0, 0.0, L.ROLLER_Z), L.ROLLER_R, L.ROLLER_LEN, axis="X", color=L.ROLLER_STEEL),
        ],
    )


SUPPORTS = [_support("left"), _support("right")]
MAT = Box("specimen_mat", pos=(L.MAT_X, L.MAT_Y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE, color=L.MAT_BLACK)
SPECIMEN = Box("specimen", pos=(L.SP_START[0], L.SP_START[1], L.TOP + L.SP_START[2]), size=L.SP_SIZE,
               mass=L.SP_MASS, color=L.SPECIMEN_CREAM)
OBJECTS = [BASE, CENTRE_LINE, *TICKS, *SUPPORTS, MAT, SPECIMEN]

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


def _parts(objects):
    return objects["specimen"], (objects["support_left"], objects["support_right"])


def success(objects) -> bool:
    """Both supports standing on the base with their rollers along the centre line, 160 mm apart within 5 mm and
    centred on the line within 5 mm; the specimen flat across both rollers, its middle within 5 mm of the span's
    middle and square to the rollers within 3 degrees; everything at rest (no end of the specimen and no roller moving
    more than 2 mm in 0.5 s) and both grippers' finger zones 20 mm or more from all three; held 1 s. This runs first
    every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    sp, sups = _parts(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(sp, sups, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    try:
        a, b = sp_ends(sp)
        row.update(sources=[getattr(s, "source", None) for s in sups], sp_pos=sp.pos, sp_quat=sp.quat_xyzw,
                   sp_vel=sp.lin_vel, end_a=a, end_b=b, support_left_origin=sups[0].origin,
                   support_left_quat=sups[0].quat_xyzw, support_right_origin=sups[1].origin,
                   support_right_quat=sups[1].quat_xyzw, roller_left=roller_centre(sups[0]),
                   roller_right=roller_centre(sups[1]))
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
    """0 unless, at the end, everything is at rest and let go (both grippers' finger zones 20 mm or more from the
    specimen and both supports), both supports stand upright on the base (within 5 degrees, the middle of the
    underside within 2 mm of its top and inside its outline) with their rollers within 5 degrees of the centre line's
    direction, the specimen lies flat (within 5 degrees) across both rollers (within 10 degrees of square to each, its
    underside within 2 mm of each roller's top, its full width on each roller, each roller at least 5 mm in from its
    end), and the span is within 10 mm of 160 mm with its middle within 8 mm of the centre line. Then the span scores
    0.5 (within 5 mm, middle within 5 mm) or 0.25, plus 0.5 if the specimen's middle is within 5 mm of the span's
    middle and it is square within 3 degrees, 0.25 if within 10 mm, else 0. Read from the state at hand (the last
    step)."""
    sp, sups = _parts(objects)
    return _TRACKER.result(sp, sups, rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
SUP_OPEN = -0.025  # 50 mm: 10 mm either side of a block's 30 mm
SUP_GRIP = -(L.BLOCK_SIZE[1] - 0.008) / 2  # 22 mm commanded across the 30 mm
SP_OPEN = -0.025  # 50 mm: 15 mm either side of the bar's 20 mm
SP_GRIP = -(L.SP_W - 0.008) / 2  # 12 mm commanded across the 20 mm
# Let go of the bar opening to 40 mm, not 50: straight up over the span the arm reaches only 12 cm (joint4), and there
# the finger zone of a 40 mm opening is 23 mm clear of the bar (a 50 mm opening would leave 18 mm, under the 20 mm).
SP_RELEASE = -0.020
# Each arm moves the support on its side, gripped across the block (yaw 0: the fingers close along y); the left arm
# then lays the specimen, gripped at its middle across its width (yaw -pi/2: the fingers close along x; +pi/2 runs
# joint6 into its limit over the fixture). Nothing is turned in the hand: the rollers stay along x and the bar along y.
YAW_SUP, YAW_SP = 0.0, -math.pi / 2
HOVER = L.TOP + 0.12  # over the parts (the blocks' tops are 43 mm up); the bar is carried with its underside 104 mm up
APPROACH = 0.030  # each final descent starts this far above its end height
SUP_GRIP_UP = 0.019  # the TCP 19 mm above a block's underside: pad tips 6 mm above it, the pads on its whole side
SUP_LIFT = L.TOP + 0.07  # a support is moved with its underside 41 mm above the base
SUP_DROP = 0.001  # a support is set down with its underside 1 mm above the base (the arm sags about 3 mm under it)
SP_GRIP_UP = 0.016  # the TCP 16 mm above the bar's underside: pad tips 3 mm above it (and above the mat)
SP_DROP = 0.001  # the bar is let go with its underside 1 mm above the rollers' tops
# Trims for the GPU run. The compliance model (J K^-1 J^T of the load's weight, joints 80/80/80/10/10/10 N m/rad,
# yamkin) puts a gripper holding a 0.13 kg support at its place pose (-1.4, +-1.0, -3.0) mm off (the +-1.0 mm outward,
# away from the midline) and the left gripper holding the 0.057 kg bar at (0.31, 0) (-0.4, +0.4, -1.4) mm off. Each
# place point is moved by about half the sideways part, so the span lands within about 1 mm of 160 mm and the bar
# within about 0.5 mm of the span's middle with the sag or without it (the offline check). If span_mm or
# sp_offset_mm in TASK_TRACE are off by more than 2 mm, move SUP_TRIM (half the span error) or SP_TRIM_Y.
SUP_TRIM = 0.0005  # each support placed 0.5 mm inside its nominal 80 mm
SP_TRIM_Y = -0.0002
SUP_PLACE_Y = L.SPAN_NOMINAL / 2 - SUP_TRIM
PARK = {"left": (0.20, 0.23, L.TOP + 0.16), "right": (0.20, -0.26, L.TOP + 0.16)}  # aside, clear of everything
# The clock: the left arm moves its support first while the right arm waits aside; the right arm moves its own once the
# left gripper has risen from the fixture (RIGHT_GO), while the left arm picks up the specimen; the left arm holds the
# specimen up over the mat until the right gripper has risen from its support (SP_GO), then lays it on the rollers.
# Two grippers are never over the fixture at the same time.
RIGHT_GO = 14.5
SP_GO = 29.0


def _support_moves(side: str) -> list[Waypoint]:
    """One arm: grip its support where it stands, lift it, move it outward along the base to 80 mm from the centre
    line, set it down, let go, go straight up. 14.5 s."""
    sx, sy = L.SUP_START[side]
    py = SUP_PLACE_Y if side == "left" else -SUP_PLACE_Y
    pick_z = L.BASE_TOP_W + SUP_GRIP_UP
    place_z = L.BASE_TOP_W + SUP_DROP + SUP_GRIP_UP
    return [
        Waypoint(2.0, (sx, sy, HOVER), SUP_OPEN, f"over the {side} support", YAW_SUP),
        Waypoint(1.0, (sx, sy, pick_z + APPROACH), SUP_OPEN, "down to just above it", YAW_SUP),
        Waypoint(1.5, (sx, sy, pick_z), SUP_OPEN, "slowly down round the block", YAW_SUP),
        Waypoint(0.5, (sx, sy, pick_z), SUP_OPEN, "settle", YAW_SUP),
        Waypoint(1.5, (sx, sy, pick_z), SUP_GRIP, "grip the support", YAW_SUP),
        Waypoint(1.0, (sx, sy, SUP_LIFT), SUP_GRIP, "lift it off the base", YAW_SUP),
        Waypoint(1.5, (sx, py, SUP_LIFT), SUP_GRIP, "move it out to 80 mm from the centre line", YAW_SUP),
        Waypoint(0.5, (sx, py, SUP_LIFT), SUP_GRIP, "settle", YAW_SUP),
        Waypoint(2.0, (sx, py, place_z), SUP_GRIP, "slowly set it on the base", YAW_SUP),
        Waypoint(0.5, (sx, py, place_z), SUP_GRIP, "settle", YAW_SUP),
        Waypoint(1.5, (sx, py, place_z), SUP_OPEN, "let go of the support", YAW_SUP),
        Waypoint(1.0, (sx, py, HOVER), SUP_OPEN, "straight up", YAW_SUP),
    ]


def _left() -> list[Waypoint]:
    """Left arm: move the left support out to 80 mm, then pick the specimen off the mat (gripped at its middle across
    its width), hold it up until the right gripper has risen, lay it flat across the rollers centred on the span, let
    go, go up and aside."""
    moves = _support_moves("left")
    px, py = L.SP_START[0], L.SP_START[1]
    pick_z = L.TOP + L.MAT_TOP + SP_GRIP_UP
    qx, qy = L.BASE_X, L.MID_Y + SP_TRIM_Y
    qz = L.ROLLER_TOP_W + SP_DROP + SP_GRIP_UP
    moves += [
        Waypoint(2.0, (px, py, HOVER), SP_OPEN, "over the specimen's middle", YAW_SP),
        Waypoint(1.0, (px, py, pick_z + APPROACH), SP_OPEN, "down to just above it", YAW_SP),
        Waypoint(1.5, (px, py, pick_z), SP_OPEN, "slowly down round the bar", YAW_SP),
        Waypoint(0.5, (px, py, pick_z), SP_OPEN, "settle", YAW_SP),
        Waypoint(1.5, (px, py, pick_z), SP_GRIP, "grip the specimen across its width", YAW_SP),
        Waypoint(1.5, (px, py, HOVER), SP_GRIP, "lift it off the mat", YAW_SP),
    ]
    t = sum(w.duration for w in moves)
    moves += [
        Waypoint(SP_GO - t, (px, py, HOVER), SP_GRIP, "hold it until the right gripper is up", YAW_SP),
        Waypoint(2.5, (qx, qy, HOVER), SP_GRIP, "carry it over the middle of the span", YAW_SP),
        Waypoint(0.8, (qx, qy, HOVER), SP_GRIP, "settle over the span", YAW_SP),
        Waypoint(1.0, (qx, qy, qz + APPROACH), SP_GRIP, "down to 3 cm above the rollers", YAW_SP),
        Waypoint(2.0, (qx, qy, qz), SP_GRIP, "slowly lower it onto the rollers", YAW_SP),
        Waypoint(0.5, (qx, qy, qz), SP_GRIP, "settle", YAW_SP),
        Waypoint(1.5, (qx, qy, qz), SP_RELEASE, "let go of the specimen", YAW_SP),
        Waypoint(0.5, (qx, qy, qz), SP_RELEASE, "settle", YAW_SP),
        Waypoint(1.0, (qx, qy, HOVER), SP_RELEASE, "straight up", YAW_SP),
        Waypoint(1.5, PARK["left"], SP_RELEASE, "aside", YAW_SP),
    ]
    return moves


def _right() -> list[Waypoint]:
    """Right arm: wait aside until the left gripper has risen from the fixture, move the right support out to 80 mm,
    go up and aside."""
    return [
        Waypoint(2.0, PARK["right"], SUP_OPEN, "aside while the left arm works", YAW_SUP),
        Waypoint(RIGHT_GO - 2.0, PARK["right"], SUP_OPEN, "wait until the left gripper is up", YAW_SUP),
        *_support_moves("right"),
        Waypoint(1.5, PARK["right"], SUP_OPEN, "aside", YAW_SUP),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
