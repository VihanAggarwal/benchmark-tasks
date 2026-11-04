"""On the station table: a yellow U of three fence walls (10 mm thick, 60 mm tall, fixed) standing in for a robot
cell, open at the front and the top, its floor x 26..36 cm, y -9..+9 cm, with a red stripe plate on the table marking
the front boundary; a loose blue aluminium block (30 mm cube, 70 g), the dropped part, lying on the cell floor; a
yellow hold-to-run enabling pad (a fixed 40 mm cylinder, 20 mm tall) on the table at the left, out of the right
arm's reach; and a grey recovery tray (80 x 140 mm inside, 15 mm walls, fixed) at the right, out of the left arm's
reach. Rule: whenever any gripper is inside the cell (its fingers inside the cell's footprint, lower than 30 cm
above the table), the left gripper must be holding the pad down on that same step; the part goes to the tray; and
the pad must be released at the end. The scorer checks it at every control step, not only at the end.

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
# Still for 1 s with the part in the tray, the pad released and both grippers let go of the part (each open,
# opening >= 0.9, or its finger zone > 10 mm from the part). Success needs the gripper let go, so the trial cannot
# end while a gripper still holds the part, whatever the hold time.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Recover the blue block from inside the yellow fenced cell and put it in the grey tray. Whenever a gripper is "
    "inside the cell, the left gripper must be holding the yellow enabling pad down; release the pad once the cell "
    "is clear."
)

_wz = L.TOP + L.WALL_HEIGHT / 2
CELL = [  # three fixed walls, open at the front (the x 0.25 side) and the top; the table is the cell's floor
    Box("cell_wall_back", pos=(L.BACK_WALL_X, 0.0, _wz), size=(L.WALL_THICK, L.BACK_WALL_LEN, L.WALL_HEIGHT),
        color=L.FENCE_YELLOW),
    Box("cell_wall_left", pos=(L.SIDE_WALL_X, L.SIDE_WALL_Y, _wz), size=(L.SIDE_WALL_LEN, L.WALL_THICK, L.WALL_HEIGHT),
        color=L.FENCE_YELLOW),
    Box("cell_wall_right", pos=(L.SIDE_WALL_X, -L.SIDE_WALL_Y, _wz), size=(L.SIDE_WALL_LEN, L.WALL_THICK, L.WALL_HEIGHT),
        color=L.FENCE_YELLOW),
]
STRIPE = Box("front_stripe", pos=(L.STRIPE_X, 0.0, L.TOP + L.STRIPE_SIZE[2] / 2), size=L.STRIPE_SIZE,
             color=L.STRIPE_RED)
PART = Box("part", pos=(L.PART_X, L.PART_Y, L.TOP + L.PART_SIZE[2] / 2), size=L.PART_SIZE, mass=L.PART_MASS,
           color=L.PART_BLUE)
PAD = Cylinder("enabling_pad", pos=(L.PAD_X, L.PAD_Y, L.TOP + L.PAD_HEIGHT / 2), radius=L.PAD_RADIUS,
               height=L.PAD_HEIGHT, color=L.PAD_YELLOW)


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
OBJECTS = [*CELL, STRIPE, PART, PAD, *TRAY]

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
    """Part resting in the tray, the pad released, both grippers let go of the part, no step on which a gripper was
    in the cell with the pad not pressed; held still 1 s. This runs first every control step, so it also feeds the
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
    """0 if, on any step, a gripper's fingers were inside the cell (footprint x 26..36 cm, y -9..+9 cm, below 30 cm
    up) while the left gripper was not pressing the pad, or if the part ends off the table. Else 1.0 part resting in
    the tray, pad released, both grippers let go of the part; 0.75 part in the tray but the pad still pressed or a
    gripper still holding the part; 0.5 part's centre out of the cell but not in the tray; 0.25 part lifted at some
    step with the pad pressed but its centre still in the cell; 0 otherwise."""
    return _TRACKER.result(objects["part"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
CLOSED = -0.008  # the fingers 16 mm apart, the narrowest commanded gap: pressing the pad with closed fingers
GRIP = -(L.PART_SIZE[1] - 0.008) / 2  # across the part's 30 mm (22 mm)
# At yaw 0 the fingers close along y (the references' grips at yaw +-pi/2 close along x). Closing along y, the open
# fingers (94 mm apart, 110 mm over the pads) span y -3.5..+7.5 cm about the part, 15 mm clear of the cell's side
# walls (inner faces y +-9 cm), and fit the tray's 140 mm along y with 15 mm to each end wall.
YAW = 0.0
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.004  # the part is gripped with the finger tips 4 mm above its bottom: 26 mm of its 30 mm held
PART_TCP_Z = L.TOP + GRIP_UP + PAD_TIPS  # tips 4 mm above the cell floor as the fingers close on the part
SET_TCP_Z = L.TOP + 0.002 + GRIP_UP + PAD_TIPS  # let go with the part's bottom 2 mm above the tray's floor
APPROACH = 0.030  # the final descent onto the part starts this far above the grip height
ENTRY = 0.020  # the last 20 mm into the tray are lowered slowly
PRESS_TCP_Z = L.PAD_TOP_Z + 0.002 + PAD_TIPS  # closed finger tips 2 mm above the pad's top (the unloaded arm's
#                                               sag brings them onto it); 'pressed' allows 5 mm below to 10 mm above
PRESS_ENTRY = 0.020  # the last 20 mm onto the pad are lowered slowly
HOVER_L = L.TOP + 0.10  # the left over the pad: tips 67 mm above the pad's top, well past the 30 mm 'released'
HOVER_R = L.TOP + 0.12  # the right's carrying height: tips at 10.7 cm, inside the 30 cm cell volume, and a carried
#                         part's bottom 10.3 cm up, over the 6 cm fence; the reach at the midline shrinks with height
#                         (x 0.40 at the table, 0.30 at 16 cm up), so the part at x 31 cm is taken low
WAIT_X, WAIT_Y = L.PART_X, -0.190  # the right waits here, outside the cell: its open finger zone (57 mm) stays
#                                    43 mm clear of the footprint's y -9 cm edge, room for an overshoot on the way
PRESS_MARGIN_S = 1.0  # the right enters the cell this long after the left's press is complete (the spec's 0.5 s
#                       plus 0.5 s for the arm's tracking lag)
EXIT_MARGIN_S = 0.5  # the left keeps the pad pressed this long after the right's TCP reaches the tray hover


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left_press() -> list[Waypoint]:
    px, py = L.PAD_X, L.PAD_Y
    return [
        Waypoint(1.5, (px, py, HOVER_L), OPEN, "over the enabling pad", YAW),
        Waypoint(1.5, (px, py, HOVER_L), CLOSED, "close the fingers", YAW),
        Waypoint(1.0, (px, py, PRESS_TCP_Z + PRESS_ENTRY), CLOSED, "down to 2 cm above the pad", YAW),
        Waypoint(1.5, (px, py, PRESS_TCP_Z), CLOSED, "press the pad (last 2 cm slowly)", YAW),
    ]


PRESSED_S = _span(_left_press())  # the left's finger tips are down on the pad


def _right() -> list[Waypoint]:
    """Right arm: wait outside the cell until the pad is pressed, go in over the part, grip it, lift it straight up,
    carry it out to the tray, set it down, let go, go up."""
    cx, cy = L.PART_X, L.PART_Y
    tx, ty = L.TRAY_X, L.TRAY_Y
    approach = [Waypoint(1.5, (WAIT_X, WAIT_Y, HOVER_R), OPEN, "wait outside the cell, beside its open front", YAW)]
    wait = PRESSED_S + PRESS_MARGIN_S - _span(approach)
    return approach + [
        Waypoint(wait, (WAIT_X, WAIT_Y, HOVER_R), OPEN, "wait for the pad to be pressed", YAW),
        Waypoint(1.5, (cx, cy, HOVER_R), OPEN, "into the cell, over the part", YAW),
        Waypoint(1.0, (cx, cy, PART_TCP_Z + APPROACH), OPEN, "down to 3 cm above the grip", YAW),
        Waypoint(1.5, (cx, cy, PART_TCP_Z), OPEN, "slowly down beside the part", YAW),
        Waypoint(0.5, (cx, cy, PART_TCP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (cx, cy, PART_TCP_Z), GRIP, "grip the part", YAW),
        Waypoint(0.5, (cx, cy, PART_TCP_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.5, (cx, cy, HOVER_R), GRIP, "lift it straight up", YAW),
        Waypoint(2.0, (tx, ty, HOVER_R), GRIP, "carry it out of the cell, over the tray", YAW),
        Waypoint(1.0, (tx, ty, SET_TCP_Z + ENTRY), GRIP, "down to 2 cm above the set-down", YAW),
        Waypoint(0.5, (tx, ty, SET_TCP_Z + ENTRY), GRIP, "settle", YAW),
        Waypoint(1.5, (tx, ty, SET_TCP_Z), GRIP, "lower the last 2 cm slowly", YAW),
        Waypoint(0.5, (tx, ty, SET_TCP_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (tx, ty, SET_TCP_Z), OPEN, "let go", YAW),
        Waypoint(1.0, (tx, ty, HOVER_R), OPEN, "up", YAW),
    ]


AT_TRAY_S = _span(_right()[:10])  # the right's TCP is over the tray: its fingers left the cell during that move


def _left() -> list[Waypoint]:
    """Left arm: close the fingers over the pad, press it, hold it pressed until the right gripper is out of the cell
    and over the tray plus a margin, then lift off (release) and stay up."""
    px, py = L.PAD_X, L.PAD_Y
    press = _left_press()
    hold = AT_TRAY_S + EXIT_MARGIN_S - _span(press)
    return press + [
        Waypoint(hold, (px, py, PRESS_TCP_Z), CLOSED, "hold the pad pressed while the right gripper is inside", YAW),
        Waypoint(1.0, (px, py, HOVER_L), CLOSED, "release the pad: up", YAW),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
