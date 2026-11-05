"""On the station table, left of the midline: a grey square sleeve (30 x 30 mm inside, 80 mm tall, fixed) with a
black enabling plunger (a 24 mm x 120 mm cylinder, 0.10 kg) standing in it on the table, and a scale post beside it
(red, green, red bands: the plunger's top is level with the green band when it is lifted 15 to 30 mm). Right of
the midline: a light grey stage (a 200 x 200 mm plate with a 20 mm frame round it, fixed) with a white 40 mm
carriage on it at the green home mark, and red and blue marks on the frame. Hand-guide the carriage to the red
mark, then the blue mark, stopping 1 s on each, then back home; it may move only while the plunger is held upright
in its middle band (lifted 15 to 30 mm, not dropped, not pulled higher), never faster than 5 cm/s. Finish with the
plunger down and both grippers open. Only the left arm reaches the plunger, only the right arm can grip the
carriage.

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
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with the carriage home, the plunger down and both grippers open and clear.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Hand-guide the white carriage, never more than 5 mm above its plate, to the red mark, then to the blue mark, "
    "stopping at least 1 s on each, then back to the green home mark. It may move only while the black enabling "
    "plunger on the left is held upright in its middle band (lifted 15 to 30 mm: not dropped, not pulled higher), "
    "and "
    "never faster than 5 cm per second. Finish with the plunger down and both grippers open."
)


def _walls(prefix: str, x: float, y: float, inner_x: float, inner_y: float, wall: float, z0: float, height: float,
           color) -> list[Box]:
    """Four fixed walls round an ``inner_x`` x ``inner_y`` opening centred at (x, y), standing on height ``z0``;
    the front and back walls (along y) run the full outer width, the side walls (along x) fit between them."""
    z = z0 + height / 2
    ox, oy = inner_x / 2 + wall / 2, inner_y / 2 + wall / 2
    return [
        Box(f"{prefix}_wall_front", pos=(x + ox, y, z), size=(wall, inner_y + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_back", pos=(x - ox, y, z), size=(wall, inner_y + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_left", pos=(x, y + oy, z), size=(inner_x, wall, height), color=color),
        Box(f"{prefix}_wall_right", pos=(x, y - oy, z), size=(inner_x, wall, height), color=color),
    ]


# -- the enabling station -------------------------------------------------------------------------------------------
SLEEVE = _walls("sleeve", L.SLEEVE_X, L.SLEEVE_Y, L.SLEEVE_INNER, L.SLEEVE_INNER, L.SLEEVE_WALL, L.TOP,
                L.SLEEVE_HEIGHT, L.GREY)
PLUNGER = Cylinder(
    "enable_plunger",
    pos=(L.SLEEVE_X, L.SLEEVE_Y, L.TOP + L.PLUNGER_HEIGHT / 2),
    radius=L.PLUNGER_RADIUS,
    height=L.PLUNGER_HEIGHT,
    mass=L.PLUNGER_MASS,
    color=L.BLACK,
)
POST = [
    Box("scale_post", pos=(L.POST_X, L.POST_Y, L.TOP + L.POST_WHITE_TOP / 2),
        size=(L.POST_WIDTH, L.POST_WIDTH, L.POST_WHITE_TOP), color=L.WHITE),
    *[
        Box(f"scale_band_{i}_{name}",
            pos=(L.POST_X, L.POST_Y, L.TOP + L.POST_WHITE_TOP + (i + 0.5) * L.BAND_HEIGHT),
            size=(L.BAND_WIDTH, L.BAND_WIDTH, L.BAND_HEIGHT), color=color)
        for i, (name, color) in enumerate((("red", L.RED_C), ("green", L.GREEN_C), ("red", L.RED_C)))
    ],
]

# -- the stage --------------------------------------------------------------------------------------------------
PLATE = Box("stage_plate", pos=(L.PLATE_X, L.PLATE_Y, L.TOP + L.PLATE_THICK / 2),
            size=(L.PLATE_SIZE, L.PLATE_SIZE, L.PLATE_THICK), color=L.STAGE_GREY)
FRAME = _walls("frame", L.PLATE_X, L.PLATE_Y, 2 * L.FRAME_INNER_HALF, 2 * L.FRAME_INNER_HALF, L.FRAME_WALL,
               L.PLATE_TOP_Z, L.FRAME_HEIGHT, L.STAGE_GREY)
_WALL_MID = L.FRAME_INNER_HALF + L.FRAME_WALL / 2  # from the plate's centre to each wall's centre line
_MARK_Z = L.FRAME_TOP_Z + L.MARK_THICK / 2


def _marks(name: str, color) -> list[Box]:
    """Four 10 x 10 mm squares on the frame's top: on both side walls at the mark's x, on the front and back walls
    at its y; the mark is where the lines joining opposite squares cross."""
    tx, ty = L.TARGETS[name]
    size = (L.MARK_SIZE, L.MARK_SIZE, L.MARK_THICK)
    return [
        Box(f"mark_{name}_left", pos=(tx, L.PLATE_Y + _WALL_MID, _MARK_Z), size=size, color=color),
        Box(f"mark_{name}_right", pos=(tx, L.PLATE_Y - _WALL_MID, _MARK_Z), size=size, color=color),
        Box(f"mark_{name}_front", pos=(L.PLATE_X + _WALL_MID, ty, _MARK_Z), size=size, color=color),
        Box(f"mark_{name}_back", pos=(L.PLATE_X - _WALL_MID, ty, _MARK_Z), size=size, color=color),
    ]


MARKS = [*_marks("red", L.RED_C), *_marks("blue", L.BLUE_C), *_marks("home", L.GREEN_C)]
CARRIAGE = Box(
    "carriage",
    pos=(L.HOME[0], L.HOME[1], L.PLATE_TOP_Z + L.CARRIAGE_HALF),
    size=(L.CARRIAGE_SIZE, L.CARRIAGE_SIZE, L.CARRIAGE_SIZE),
    mass=L.CARRIAGE_MASS,
    color=L.WHITE,
)
OBJECTS = [*SLEEVE, PLUNGER, *POST, PLATE, *FRAME, *MARKS, CARRIAGE]

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
    """Stopped (1 s, enabled) at red, then blue, then home; the carriage within 5 mm of home, the plunger down, both
    grippers open and clear; never moved while not enabled, never faster than 5 cm/s; held still 1 s. This runs
    first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    carriage, plunger = objects["carriage"], objects["enable_plunger"]
    arms = rigstate.arms()
    result = _TRACKER.update(carriage, plunger, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "carriage_pos": carriage.pos, "plunger_pos": plunger.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
        try:  # for tuning CARRIAGE_AIM: where the carriage sits relative to the right TCP while it is guided
            rt = arms["right"].tcp
            row["carriage_minus_right_tcp_mm"] = [round(1000 * (float(carriage.pos[i]) - float(rt[i])), 1)
                                                  for i in (0, 1)]
        except (TypeError, ValueError, IndexError):
            pass
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, the carriage travelled more than 2 mm in all while the plunger was not held upright
    (under 4 degrees) in its band (15-30 mm up), moved faster than 50 mm/s, was lifted more than 5 mm off the plate
    or tilted over 10 degrees, or the plunger tilted over 15 degrees, left its sleeve, or anything left the table.
    Else 1.0 stopped 1 s (enabled, within 3 mm) at red, then blue, then home, carriage within 5 mm of home, plunger
    down, both grippers open and 30 mm clear; 0.75 red then blue; 0.5 red; 0.25 the carriage 20 mm from where one
    enabled stretch of 3 s began; 0 otherwise."""
    return _TRACKER.result(objects["carriage"], objects["enable_plunger"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PLUNGER_GRIP = -(2 * L.PLUNGER_RADIUS - 0.008) / 2  # 16 mm on the 24 mm plunger
CARRIAGE_GRIP = -(L.CARRIAGE_SIZE - 0.008) / 2  # 32 mm on the 40 mm carriage
PRECLOSE = -0.035  # 70 mm: the right fingers come down 15 mm clear of the carriage's sides, so the close is short
# Left yaw -90 deg: the fingers close along x, so the open pads spread along x and miss the scale post (+43 mm in y).
# Right yaw +90 deg: the fingers close along x, across the carriage, clear of the frame walls at every mark.
YAW_L = -math.pi / 2
YAW_R = math.pi / 2
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP

PX, PY = L.SLEEVE_X, L.SLEEVE_Y
P_GRIP_Z = L.TOP + L.PLUNGER_HEIGHT - L.PLUNGER_GRIP_BELOW_TOP  # pad tips 7 mm above the sleeve's top
P_ENTRY_Z = P_GRIP_Z + 0.020  # the last 20 mm are lowered slowly
P_HOVER_Z = L.TOP + 0.140
P_LIFT = 0.0225  # the plunger lifted 22.5 mm: the middle of its 15-30 mm band
P_UP_Z = L.TOP + 0.170  # at the end, the pad tips 37 mm above the plunger's top

C_GRIP_Z = L.PLATE_TOP_Z + 0.009 + PAD_TIPS  # pad tips 9 mm above the plate: 22 mm of the 40 mm side held
C_ENTRY_Z = L.TOP + 0.060  # the last 32 mm are lowered slowly
C_HOVER_Z = L.TOP + 0.120
C_LIFT = 0.003  # the carriage guided 3 mm off the plate, so it slides on nothing (the arm sags ~2 mm under it)
C_UP_Z = L.TOP + 0.100  # at the end, the pad tips 41 mm above the carriage's top
# Added to every point the carriage is guided to, to cancel a steady offset between the TCP and the gripped
# carriage seen on the GPU (TASK_TRACE: carriage_minus_right_tcp_mm). Zero offline.
CARRIAGE_AIM = (0.0, 0.0)


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left_enable() -> list[Waypoint]:
    p = (PX, PY)
    return [
        Waypoint(3.0, (*p, P_HOVER_Z), OPEN, "over the plunger", YAW_L),
        Waypoint(2.0, (*p, P_ENTRY_Z), OPEN, "down beside it", YAW_L),
        Waypoint(1.5, (*p, P_GRIP_Z), OPEN, "the last 2 cm slowly", YAW_L),
        Waypoint(0.5, (*p, P_GRIP_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (*p, P_GRIP_Z), PLUNGER_GRIP, "grip the plunger", YAW_L),
        Waypoint(0.5, (*p, P_GRIP_Z), PLUNGER_GRIP, "settle", YAW_L),
        Waypoint(1.5, (*p, P_GRIP_Z + P_LIFT), PLUNGER_GRIP, "lift it into its middle band (enabled)", YAW_L),
    ]


ENABLED_S = _span(_left_enable())  # the plunger is up in its band (enabled from about 0.4 s before this)


def _aim(xy) -> tuple[float, float]:
    return xy[0] + CARRIAGE_AIM[0], xy[1] + CARRIAGE_AIM[1]


def _right() -> list[Waypoint]:
    """Right arm: come down round the carriage, wait until the plunger is held in its band, grip the carriage,
    guide it 3 mm above the plate to red, blue and home at under 32 mm/s, stopping 2.5 s on each (the
    scorer needs 1 s once it has settled), set it down, let go and go up."""
    home, red, blue = L.HOME, _aim(L.RED), _aim(L.BLUE)
    g = C_GRIP_Z + C_LIFT
    approach = [
        Waypoint(3.0, (*home, C_HOVER_Z), PRECLOSE, "over the carriage", YAW_R),
        Waypoint(2.0, (*home, C_ENTRY_Z), PRECLOSE, "down over it", YAW_R),
        Waypoint(1.5, (*home, C_GRIP_Z), PRECLOSE, "the last 3 cm slowly, beside it", YAW_R),
    ]
    wait = ENABLED_S + 0.5 - _span(approach)
    return approach + [
        Waypoint(wait, (*home, C_GRIP_Z), PRECLOSE, "wait until the plunger is held in its band", YAW_R),
        Waypoint(1.5, (*home, C_GRIP_Z), CARRIAGE_GRIP, "grip the carriage", YAW_R),
        Waypoint(0.5, (*home, C_GRIP_Z), CARRIAGE_GRIP, "settle", YAW_R),
        Waypoint(1.0, (*_aim(home), g), CARRIAGE_GRIP, "lift it 3 mm off the plate", YAW_R),
        Waypoint(2.5, (*red, g), CARRIAGE_GRIP, "guide it to the red mark", YAW_R),
        Waypoint(2.5, (*red, g), CARRIAGE_GRIP, "stop on the red mark", YAW_R),
        Waypoint(5.0, (*blue, g), CARRIAGE_GRIP, "guide it to the blue mark", YAW_R),
        Waypoint(2.5, (*blue, g), CARRIAGE_GRIP, "stop on the blue mark", YAW_R),
        Waypoint(3.0, (*_aim(home), g), CARRIAGE_GRIP, "guide it back home", YAW_R),
        Waypoint(2.5, (*_aim(home), g), CARRIAGE_GRIP, "stop at home", YAW_R),
        Waypoint(1.0, (*_aim(home), C_GRIP_Z), CARRIAGE_GRIP, "set it down on the plate", YAW_R),
        Waypoint(0.5, (*_aim(home), C_GRIP_Z), CARRIAGE_GRIP, "settle", YAW_R),
        Waypoint(1.5, (*_aim(home), C_GRIP_Z), OPEN, "let go", YAW_R),
        Waypoint(0.5, (*_aim(home), C_GRIP_Z), OPEN, "settle", YAW_R),
        Waypoint(1.5, (*_aim(home), C_UP_Z), OPEN, "up, clear of the carriage", YAW_R),
    ]


RIGHT_CLEAR_S = _span(_right())  # the right gripper is open and up: the carriage is done


def _left() -> list[Waypoint]:
    """Left arm: lift the plunger into its band and hold it there steady until the right arm has let go of the
    carriage and gone up, then lower it, let go and go up."""
    enable = _left_enable()
    up = (PX, PY, P_GRIP_Z + P_LIFT)
    return enable + [
        Waypoint(RIGHT_CLEAR_S - ENABLED_S, up, PLUNGER_GRIP, "hold it in the band", YAW_L),
        Waypoint(1.5, (PX, PY, P_GRIP_Z), PLUNGER_GRIP, "lower it back down", YAW_L),
        Waypoint(0.5, (PX, PY, P_GRIP_Z), PLUNGER_GRIP, "settle", YAW_L),
        Waypoint(1.5, (PX, PY, P_GRIP_Z), OPEN, "let go", YAW_L),
        Waypoint(0.5, (PX, PY, P_GRIP_Z), OPEN, "settle", YAW_L),
        Waypoint(2.0, (PX, PY, P_UP_Z), OPEN, "up", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
