"""On the station table: a press nest (four fixed dark grey steel rails, 8 mm thick and 8 mm tall, round a 36 x 36 mm
floor) on the midline 30 cm in front of the arm bases, with a fixed grey press frame (10 x 160 x 120 mm) standing
just behind it; a loose blue aluminium block (30 mm cube, 70 g), the part, standing on a thin white feed mark plate
at x 24 cm, y +12 cm (in reach of both arms); two yellow run buttons (fixed 40 mm cylinders, 20 mm tall) at the far
left (x 20 cm, y +21 cm, out of the right arm's reach) and the far right (x 20 cm, y -21 cm, out of the left arm's
reach); and a grey output tray (80 x 140 mm inside, 15 mm walls, fixed) at x 16 cm on the midline. Two-hand control
rule: the press cycle runs only while both run buttons are held at once, pressed within 0.5 s of each other and held
2 s with the part seated in the nest; no gripper may be in the die space while any button is pressed. Load the part,
run the cycle, unload it into the tray. The scorer checks it at every control step, not only at the end.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
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
# Still for 1 s with the cycled part in the tray, both buttons released and both grippers let go of the part (each
# open, opening >= 0.9, or its finger zone > 10 mm from the part). Success needs the gripper let go, so the trial
# cannot end while a gripper still holds the part, whatever the hold time.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Load the blue block into the press nest between the rails, then run the press cycle: hold both yellow run "
    "buttons down at the same time, one with each gripper, pressing them within half a second of each other and "
    "keeping both held for two seconds. Never have a gripper near the nest while a button is pressed. Then let go of "
    "the buttons and move the block into the grey tray."
)

_rz = L.TOP + L.RAIL_HEIGHT / 2
_ro = L.NEST_INNER / 2 + L.RAIL_THICK / 2  # rail centres this far from the nest's centre
_rl = L.NEST_INNER + 2 * L.RAIL_THICK  # the two rails along y span the full outer width; the other two fit between
NEST = [  # four fixed rails; the table top is the nest's floor
    Box("rail_front", pos=(L.NEST_X + _ro, L.NEST_Y, _rz), size=(L.RAIL_THICK, _rl, L.RAIL_HEIGHT), color=L.RAIL_GREY),
    Box("rail_back", pos=(L.NEST_X - _ro, L.NEST_Y, _rz), size=(L.RAIL_THICK, _rl, L.RAIL_HEIGHT), color=L.RAIL_GREY),
    Box("rail_left", pos=(L.NEST_X, L.NEST_Y + _ro, _rz), size=(L.NEST_INNER, L.RAIL_THICK, L.RAIL_HEIGHT),
        color=L.RAIL_GREY),
    Box("rail_right", pos=(L.NEST_X, L.NEST_Y - _ro, _rz), size=(L.NEST_INNER, L.RAIL_THICK, L.RAIL_HEIGHT),
        color=L.RAIL_GREY),
]
FRAME = Box("press_frame", pos=(L.FRAME_X, L.FRAME_Y, L.TOP + L.FRAME_SIZE[2] / 2), size=L.FRAME_SIZE,
            color=L.FRAME_GREY)
PLATE = Box("feed_mark", pos=(L.PLATE_X, L.PLATE_Y, L.TOP + L.PLATE_SIZE[2] / 2), size=L.PLATE_SIZE,
            color=L.PLATE_WHITE)
PART = Box("part", pos=(L.PLATE_X, L.PLATE_Y, L.PLATE_TOP_Z + L.PART_SIZE[2] / 2), size=L.PART_SIZE,
           mass=L.PART_MASS, color=L.PART_BLUE)
BUTTONS = [
    Cylinder(f"run_button_{side}", pos=(L.BUTTON_X, L.BUTTON_Y[side], L.TOP + L.BUTTON_HEIGHT / 2),
             radius=L.BUTTON_RADIUS, height=L.BUTTON_HEIGHT, color=L.BUTTON_YELLOW)
    for side in ("left", "right")
]


def _walls(prefix: str, x: float, y: float, inner_x: float, inner_y: float, wall: float, height: float,
           color) -> list[Box]:
    """Four fixed walls around an ``inner_x`` x ``inner_y`` opening centred at (x, y); the table is the floor."""
    z = L.TOP + height / 2
    ox, oy = inner_x / 2 + wall / 2, inner_y / 2 + wall / 2
    return [
        Box(f"{prefix}_wall_front", pos=(x + ox, y, z), size=(wall, inner_y + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_back", pos=(x - ox, y, z), size=(wall, inner_y + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_left", pos=(x, y + oy, z), size=(inner_x, wall, height), color=color),
        Box(f"{prefix}_wall_right", pos=(x, y - oy, z), size=(inner_x, wall, height), color=color),
    ]


TRAY = _walls("tray", L.TRAY_X, L.TRAY_Y, L.TRAY_INNER_X, L.TRAY_INNER_Y, L.TRAY_WALL, L.TRAY_HEIGHT, L.TRAY_GREY)
OBJECTS = [*NEST, FRAME, PLATE, PART, *BUTTONS, *TRAY]

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
    """A cycle completed (both buttons pressed within 0.5 s of each other and held 2 s with the part seated), the
    part resting in the tray, both buttons released, both grippers let go of the part, no step on which a button was
    pressed with a gripper in the die space; held still 1 s. This runs first every control step, so it also feeds the
    tracker (with the arms rigstate read for this step)."""
    part = objects["part"]
    arms = rigstate.arms()
    result = _TRACKER.update(part, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "part_pos": part.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, on any step, a button was pressed while a gripper's fingers were in the die space (x 25..35 cm,
    y -5..+5 cm, below 20 cm up), or if the part ends off the table. Else 1.0 cycled part resting in the tray, both
    buttons released, both grippers let go of it; 0.75 cycled but not resting in the tray, or in the tray with a
    button still pressed or a gripper still holding it; 0.5 part seated in the nest and a double press began with it
    seated but no cycle completed; 0.25 part seated in the nest, no such double press; 0 otherwise."""
    return _TRACKER.result(objects["part"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
CLOSED = -0.008  # the fingers 16 mm apart, the narrowest commanded gap: pressing a button with closed fingers
GRIP = -(L.PART_SIZE[1] - 0.008) / 2  # across the part's 30 mm (22 mm)
# At yaw 0 the fingers close along y (the references' grips at yaw +-pi/2 close along x). Closing along y, the open
# fingers (94 mm apart, 110 mm over the pads) span y -5.5..+5.5 cm about the part, clear of the nest's rails
# (outer faces y +-2.6 cm) and 15 mm clear of the tray's end walls (inner faces y +-7 cm).
YAW = 0.0
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
RAIL_UP = 0.004  # in the nest the part is gripped with the finger tips 4 mm above the rails' tops ...
GRIP_UP = L.RAIL_HEIGHT + RAIL_UP  # ... 12 mm above its bottom: its upper 18 mm held by the pads' 22 mm faces.
# The same hold is used to pick it off the feed mark: the pads, closed on the 30 mm part, stand 8 mm proud of its
# faces, 1 mm outside the rails' inner faces, so a part held by its lower 26 mm (tips 4 mm above its bottom) could
# not be set down between the rails without the pads landing on them. Held by its upper 18 mm, it is set down with
# its bottom 2 mm above the nest's floor and the pad tips 6 mm above the rails.
READY = {"left": (0.323, 0.305, L.TOP + 0.26), "right": (0.323, -0.305, L.TOP + 0.26)}  # the arms' ready poses
HELD = PAD_TIPS + GRIP_UP - L.PART_HALF  # a held part's centre is this far below the TCP (10 mm)
FEED_TCP_Z = L.PLATE_TOP_Z + GRIP_UP + PAD_TIPS  # tips 12 mm above the feed mark as the fingers close on the part
LOAD_TCP_Z = L.TOP + 0.002 + L.PART_HALF + HELD  # let go with the part's bottom 2 mm above the nest's floor
NEST_TCP_Z = L.RAIL_TOP_Z + RAIL_UP + PAD_TIPS  # tips 4 mm above the rails' tops as the fingers close in the nest
SET_TCP_Z = L.TOP + 0.002 + L.PART_HALF + HELD  # let go with the part's bottom 2 mm above the tray's floor
APPROACH = 0.030  # the final descent onto the part starts this far above the grip height
ENTRY = 0.020  # the last 20 mm of a set-down are lowered slowly
PRESS_TCP_Z = L.BUTTON_TOP_Z + 0.002 + PAD_TIPS  # closed finger tips 2 mm above the button's top (the unloaded arm's
#                                                  sag brings them onto it); 'pressed' allows 5 mm below to 10 mm above
PRESS_ENTRY = 0.020  # the last 20 mm onto a button are lowered slowly: both arms make this same move on one clock,
#                      so their tip points enter the 'pressed' band on the same step (synchronous actuation)
PRESS_HOVER = L.TOP + 0.10  # both arms hover here over their buttons, before and after the press: tips 67 mm above
#                             the button's top, well past the 30 mm 'released'. Both press descents start from this
#                             same height, so the two arms make the same moves on the same clock.
HOVER_R = L.TOP + 0.12  # the right's carrying height: a carried part's bottom 10 cm up, over the 8 mm rails and the
#                         15 mm tray walls; the reach at the midline shrinks with height (x 0.40 at the table, 0.30
#                         at 16 cm up), so the nest at x 30 cm is worked below 16 cm
HOLD_L = 2.5  # the left holds its button this long after its press descent ends (the cycle needs 2.0 s of both)
HOLD_R = HOLD_L + 0.5  # the right holds 0.5 s longer, so the double press ends with the left's release while the
#                        right is still on its button, and the right leaves for the nest only after both are up
ASIDE_Y = 0.040  # after its release the left steps this far further left (y +25 cm), so the right's later moves on
#                  the midline (x 16 and 30 cm) pass it with 25 cm between the TCPs in y, not 21


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _right_load() -> list[Waypoint]:
    fx, fy = L.PLATE_X, L.PLATE_Y
    nx, ny = L.NEST_X, L.NEST_Y
    return [
        Waypoint(1.5, (fx, fy, HOVER_R), OPEN, "over the part on the feed mark", YAW),
        Waypoint(1.0, (fx, fy, FEED_TCP_Z + APPROACH), OPEN, "down to 3 cm above the grip", YAW),
        Waypoint(1.5, (fx, fy, FEED_TCP_Z), OPEN, "slowly down beside the part", YAW),
        Waypoint(0.5, (fx, fy, FEED_TCP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (fx, fy, FEED_TCP_Z), GRIP, "grip the part by its upper 18 mm", YAW),
        Waypoint(0.5, (fx, fy, FEED_TCP_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (fx, fy, HOVER_R), GRIP, "lift it straight up", YAW),
        Waypoint(1.5, (nx, ny, HOVER_R), GRIP, "over the nest", YAW),
        Waypoint(1.0, (nx, ny, LOAD_TCP_Z + ENTRY), GRIP, "down to 2 cm above the let-go height", YAW),
        Waypoint(0.5, (nx, ny, LOAD_TCP_Z + ENTRY), GRIP, "settle", YAW),
        Waypoint(1.5, (nx, ny, LOAD_TCP_Z), GRIP, "lower the last 2 cm slowly: the part's bottom 2 mm over the floor", YAW),
        Waypoint(0.5, (nx, ny, LOAD_TCP_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (nx, ny, LOAD_TCP_Z), OPEN, "let go in the nest", YAW),
        Waypoint(1.0, (nx, ny, HOVER_R), OPEN, "up", YAW),
    ]


AT_NEST_S = _span(_right_load()[:8])  # the right's TCP is over the nest, away from the feed mark


def _right_press_approach() -> list[Waypoint]:
    bx, by = L.BUTTON_X, L.BUTTON_Y["right"]
    return [
        Waypoint(1.5, (bx, by, PRESS_HOVER), OPEN, "over the right run button", YAW),
        Waypoint(1.5, (bx, by, PRESS_HOVER), CLOSED, "close the fingers", YAW),
    ]


PRESS_START_S = _span(_right_load()) + _span(_right_press_approach())  # both press descents start now


def _press(side: str, hold: float) -> list[Waypoint]:
    """One arm's press: from the hover, down to 2 cm above its button, the last 2 cm slowly, hold, up. Both arms play
    the same moves on the same clock, from the same height."""
    bx, by = L.BUTTON_X, L.BUTTON_Y[side]
    return [
        Waypoint(1.0, (bx, by, PRESS_TCP_Z + PRESS_ENTRY), CLOSED, "down to 2 cm above the button", YAW),
        Waypoint(1.5, (bx, by, PRESS_TCP_Z), CLOSED, "press the button (last 2 cm slowly)", YAW),
        Waypoint(hold, (bx, by, PRESS_TCP_Z), CLOSED, "hold the button down: the cycle runs", YAW),
        Waypoint(1.0, (bx, by, PRESS_HOVER), CLOSED, "release the button: up", YAW),
    ]


def _right_unload() -> list[Waypoint]:
    nx, ny = L.NEST_X, L.NEST_Y
    tx, ty = L.TRAY_X, L.TRAY_Y
    return [
        Waypoint(1.5, (nx, ny, HOVER_R), OPEN, "over the nest, opening the fingers", YAW),
        Waypoint(1.5, (nx, ny, NEST_TCP_Z), OPEN, "down beside the part, tips 4 mm above the rails", YAW),
        Waypoint(0.5, (nx, ny, NEST_TCP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (nx, ny, NEST_TCP_Z), GRIP, "grip the part by its upper 18 mm", YAW),
        Waypoint(0.5, (nx, ny, NEST_TCP_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (nx, ny, HOVER_R), GRIP, "lift it straight up out of the nest", YAW),
        Waypoint(1.5, (tx, ty, HOVER_R), GRIP, "over the output tray", YAW),
        Waypoint(1.0, (tx, ty, SET_TCP_Z + ENTRY), GRIP, "down to 2 cm above the set-down", YAW),
        Waypoint(0.5, (tx, ty, SET_TCP_Z + ENTRY), GRIP, "settle", YAW),
        Waypoint(1.5, (tx, ty, SET_TCP_Z), GRIP, "lower the last 2 cm slowly", YAW),
        Waypoint(0.5, (tx, ty, SET_TCP_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (tx, ty, SET_TCP_Z), OPEN, "let go", YAW),
        Waypoint(1.0, (tx, ty, HOVER_R), OPEN, "up", YAW),
    ]


def _right() -> list[Waypoint]:
    """Right arm: pick the part from the feed mark, load it into the nest, let go, go to its run button, close the
    fingers, press with the left on the same clock, hold, release, then unload the part from the nest to the tray."""
    return _right_load() + _right_press_approach() + _press("right", HOLD_R) + _right_unload()


def _left() -> list[Waypoint]:
    """Left arm: wait at the ready pose while the right picks from the feed mark (the two TCPs would otherwise be too
    close), then hover over its run button with the fingers closed until the right's press descent starts, press on
    the same clock, hold, release and stay up."""
    bx, by = L.BUTTON_X, L.BUTTON_Y["left"]
    wps = [
        Waypoint(AT_NEST_S, READY["left"], OPEN, "wait at the ready pose while the right arm picks the part", YAW),
        Waypoint(1.5, (bx, by, PRESS_HOVER), OPEN, "over the left run button", YAW),
        Waypoint(1.5, (bx, by, PRESS_HOVER), CLOSED, "close the fingers", YAW),
    ]
    wait = PRESS_START_S - _span(wps)
    wps.append(Waypoint(wait, (bx, by, PRESS_HOVER), CLOSED, "wait over the button until the right arm is over its own",
                        YAW))
    return wps + _press("left", HOLD_L) + [
        Waypoint(1.0, (bx, by + ASIDE_Y, PRESS_HOVER), CLOSED, "aside, out of the right arm's way", YAW),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
