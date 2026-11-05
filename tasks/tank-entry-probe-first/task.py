"""On the station table: a grey open-top steel tank (120 x 360 mm inside, 80 mm deep, 8 mm walls, fixed) on the
midline 28 cm in front of the arm bases, long side along y, with a dark 30 mm steel cube (the dropped part) on its
floor 8.5 cm right of the midline; an orange gas-monitor probe wand (a 30 mm cylinder, 150 mm long) standing in a
black holder collar to the left of the tank; and a blue parts tray to the right of it. Confined-space entry rule:
before the right (entrant) gripper reaches into the tank, the left (attendant) gripper tests the air with the probe at
the top, the middle and the bottom of the tank, in that order; it keeps the probe low in the tank the whole time the
right gripper is inside, never lets go of it there, and never goes into the tank itself.

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
# Still for 3 s with both grippers let go. Longer than the demo's last release and retreat (1.5 s + 1.0 s), so the
# trial cannot end while the left gripper is still round the probe in its holder.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Before the right gripper reaches into the grey tank, test the tank's air with the orange probe held in the left "
    "gripper: hold the probe's tip still for 2 seconds just below the rim, then at mid-depth, then just above the "
    "floor, in that order. Keep the probe in the lower half of the tank, held by the left gripper, the whole time the "
    "right gripper is inside, and never put the left gripper into the tank. With the right gripper, take the steel "
    "part out of the tank and put it in the blue tray; then put the probe back in its holder."
)


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


TANK = _walls("tank", L.TANK_X, L.TANK_Y, L.TANK_INNER_X, L.TANK_INNER_Y, L.TANK_WALL, L.TANK_DEPTH, L.TANK_GREY)
PART = Box(
    "steel_part",
    pos=(L.PART_X, L.PART_Y, L.TOP + L.PART_EDGE / 2),
    size=(L.PART_EDGE,) * 3,
    mass=L.PART_MASS,
    color=L.STEEL,
)
HOLDER = _walls("holder", L.PROBE_X, L.PROBE_Y, L.HOLDER_INNER, L.HOLDER_INNER, L.HOLDER_WALL, L.HOLDER_HEIGHT,
                L.HOLDER_BLACK)
PROBE = Cylinder(
    "probe",
    pos=(L.PROBE_X, L.PROBE_Y, L.TOP + L.PROBE_LEN / 2),
    radius=L.PROBE_RADIUS,
    height=L.PROBE_LEN,
    mass=L.PROBE_MASS,
    color=L.PROBE_ORANGE,
)
TRAY = _walls("tray", L.TRAY_X, L.TRAY_Y, L.TRAY_INNER, L.TRAY_INNER, L.TRAY_WALL, L.TRAY_HEIGHT, L.TRAY_BLUE)
OBJECTS = [*TANK, PART, *HOLDER, PROBE, *TRAY]

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
    """Part in the tray, probe upright in its holder, both grippers let go, no rule broken at any time; held still
    3 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate read)."""
    probe, part = objects["probe"], objects["steel_part"]
    arms = rigstate.arms()
    result = _TRACKER.update(probe, part, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "probe_pos": probe.pos, "part_pos": part.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if at any moment the right gripper was in the tank without a completed, still valid top-middle-bottom test
    and the probe held below half depth by the left gripper; the left gripper went into the tank; the probe was in
    the tank not held by the left gripper, was dropped outside it, or tilted over 45 degrees; the part came up out
    of the tank not held by the right gripper; or an object went off the table. Else 1.0 part in the tray, probe in
    its holder, both grippers let go; 0.75 part in the tray, probe out of the tank and held by the left gripper or
    standing on the table; 0.5 part out of the tank; 0.25 a complete test; 0 otherwise."""
    return _TRACKER.result(objects["probe"], objects["steel_part"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PROBE_GRIP = -(2 * L.PROBE_RADIUS - 0.008) / 2  # 22 mm across the probe's 30 mm
PART_GRIP = -(L.PART_EDGE - 0.008) / 2  # 22 mm across the part's 30 mm
# Both arms close the fingers along y (left yaw -pi/2, right +pi/2, the signs that reach the midline). Inside the
# tank that puts the right gripper's 110 mm open span along the tank's 360 mm length, and its 27 mm pads across the
# 120 mm width (46 mm to each long wall).
YAW_L = -math.pi / 2
YAW_R = math.pi / 2
TIPS = L.TIPS_BELOW_TCP
# The probe is gripped with the pad tips 30 mm below its top, so the TCP is 133 mm above the probe's lower end.
PROBE_TCP_ABOVE_TIP = L.PROBE_LEN - L.PROBE_GRIP_DOWN + TIPS
PROBE_PICK_Z = L.TOP + PROBE_TCP_ABOVE_TIP  # the probe standing on the table in its holder
PROBE_PLACE_Z = PROBE_PICK_Z + 0.002  # let go with its lower end 2 mm above the table, inside the collar
PROBE_CARRY_Z = L.TOP + 0.122 + PROBE_TCP_ABOVE_TIP  # lower end 42 mm above the rim, 92 mm above the collar
PROBE_HOVER_Z = L.TOP + 0.20  # over the holder, the open pad tips 37 mm above the probe's top
# Commanded tip heights above the floor: the band centres. The 0.12 kg probe sags the left arm about 3 mm (8 mm is
# the figure for 0.30 kg); every band is 20 mm tall, so a sag up to 10 mm keeps the tip inside it, and in the bottom
# band it stays 5 mm or more above the floor. The pad tips are then 55 mm above the rim.
TIP_Z = {name: (lo + hi) / 2 for name, (lo, hi) in zip(L.BAND_NAMES, L.BANDS)}  # 65, 40, 15 mm
DWELL = 3.0  # held 3 s per band: the 2 s the scorer needs and 1 s for the arm to settle after the move
PART_TCP_Z = L.TOP + 0.004 + TIPS  # pad tips 4 mm above the tank floor, on the part's y faces
PART_ENTRY_Z = PART_TCP_Z + 0.020  # the last 20 mm are lowered slowly
PART_CARRY_Z = L.TOP + 0.150  # the part's bottom 53 mm above the rim (about 47 mm with the arm's sag)
PART_WAIT_Z = L.TOP + 0.160  # the right gripper waits here, its pad tips 67 mm above the rim
DROP_TCP_Z = L.TRAY_TOP_Z + 0.015 + 0.004 + TIPS  # let go with the part's bottom 15 mm above the tray's rim
ENTRY_MARGIN_S = 0.5  # the right gripper starts down this long after the probe's bottom dwell ends
EXIT_MARGIN_S = 0.5  # the probe starts up this long after the right gripper's lift has ended


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _probe_z(name: str) -> float:
    return L.TOP + TIP_Z[name] + PROBE_TCP_ABOVE_TIP


def _left_test() -> list[Waypoint]:
    """Left arm: take the probe from its holder, carry it over the tank, and test top, middle, bottom."""
    hx, hy = L.PROBE_X, L.PROBE_Y
    sx, sy = L.SAMPLE_X, L.SAMPLE_Y
    return [
        Waypoint(1.5, (hx, hy, PROBE_HOVER_Z), OPEN, "over the probe", YAW_L),
        Waypoint(1.0, (hx, hy, PROBE_PICK_Z + 0.020), OPEN, "down to just above the grip", YAW_L),
        Waypoint(1.5, (hx, hy, PROBE_PICK_Z), OPEN, "slowly down round its top", YAW_L),
        Waypoint(0.5, (hx, hy, PROBE_PICK_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (hx, hy, PROBE_PICK_Z), PROBE_GRIP, "grip the probe", YAW_L),
        Waypoint(0.5, (hx, hy, PROBE_PICK_Z), PROBE_GRIP, "settle the grip", YAW_L),
        Waypoint(1.5, (hx, hy, PROBE_CARRY_Z), PROBE_GRIP, "lift it straight up out of the holder", YAW_L),
        Waypoint(2.0, (sx, sy, PROBE_CARRY_Z), PROBE_GRIP, "carry it over the tank", YAW_L),
        Waypoint(1.5, (sx, sy, _probe_z("top")), PROBE_GRIP, "lower the tip to just below the rim", YAW_L),
        Waypoint(DWELL, (sx, sy, _probe_z("top")), PROBE_GRIP, "test at the top", YAW_L),
        Waypoint(1.5, (sx, sy, _probe_z("middle")), PROBE_GRIP, "lower the tip to mid-depth", YAW_L),
        Waypoint(DWELL, (sx, sy, _probe_z("middle")), PROBE_GRIP, "test at mid-depth", YAW_L),
        Waypoint(1.5, (sx, sy, _probe_z("bottom")), PROBE_GRIP, "lower the tip to just above the floor", YAW_L),
        Waypoint(DWELL, (sx, sy, _probe_z("bottom")), PROBE_GRIP, "test at the bottom", YAW_L),
    ]


TEST_DONE_S = _span(_left_test())  # the end of the bottom dwell


def _right() -> list[Waypoint]:
    """Right arm: wait over the tank until the test is done, then go in, take the part, lift it out and drop it in
    the tray; go up."""
    px, py = L.PART_X, L.PART_Y
    tx, ty = L.TRAY_X, L.TRAY_Y
    approach = [Waypoint(2.0, (px, py, PART_WAIT_Z), OPEN, "over the part, pad tips well above the rim", YAW_R)]
    wait = TEST_DONE_S + ENTRY_MARGIN_S - _span(approach)
    return approach + [
        Waypoint(wait, (px, py, PART_WAIT_Z), OPEN, "wait until the probe has tested top, middle, bottom", YAW_R),
        Waypoint(1.5, (px, py, PART_ENTRY_Z), OPEN, "into the tank, down to 2 cm above the grip", YAW_R),
        Waypoint(1.5, (px, py, PART_TCP_Z), OPEN, "lower the last 2 cm slowly", YAW_R),
        Waypoint(0.5, (px, py, PART_TCP_Z), OPEN, "settle", YAW_R),
        Waypoint(1.5, (px, py, PART_TCP_Z), PART_GRIP, "grip the part", YAW_R),
        Waypoint(0.5, (px, py, PART_TCP_Z), PART_GRIP, "settle the grip", YAW_R),
        Waypoint(2.0, (px, py, PART_CARRY_Z), PART_GRIP, "lift it straight up out of the tank", YAW_R),
        Waypoint(2.0, (tx, ty, PART_CARRY_Z), PART_GRIP, "carry it over the tray", YAW_R),
        Waypoint(1.0, (tx, ty, DROP_TCP_Z), PART_GRIP, "lower it to just above the tray's rim", YAW_R),
        Waypoint(0.5, (tx, ty, DROP_TCP_Z), PART_GRIP, "settle", YAW_R),
        Waypoint(1.5, (tx, ty, DROP_TCP_Z), OPEN, "let it drop into the tray", YAW_R),
        Waypoint(1.0, (tx, ty, PART_CARRY_Z), OPEN, "up", YAW_R),
    ]


RIGHT_OUT_S = _span(_right()[:8])  # the end of the lift: the part and the right gripper are out of the tank


def _left() -> list[Waypoint]:
    """Left arm: the test, then hold the probe in the bottom band until the right gripper is out of the tank, lift it
    out, put it back in its holder, let go, go up."""
    hx, hy = L.PROBE_X, L.PROBE_Y
    sx, sy = L.SAMPLE_X, L.SAMPLE_Y
    test = _left_test()
    hold = RIGHT_OUT_S + EXIT_MARGIN_S - _span(test)
    return test + [
        Waypoint(hold, (sx, sy, _probe_z("bottom")), PROBE_GRIP, "keep monitoring low while the right is inside",
                 YAW_L),
        Waypoint(1.5, (sx, sy, PROBE_CARRY_Z), PROBE_GRIP, "lift the probe straight up out of the tank", YAW_L),
        Waypoint(2.0, (hx, hy, PROBE_CARRY_Z), PROBE_GRIP, "carry it back over the holder", YAW_L),
        Waypoint(1.0, (hx, hy, PROBE_PLACE_Z + 0.020), PROBE_GRIP, "down to 2 cm above the table", YAW_L),
        Waypoint(1.5, (hx, hy, PROBE_PLACE_Z), PROBE_GRIP, "lower the last 2 cm slowly into the collar", YAW_L),
        Waypoint(0.5, (hx, hy, PROBE_PLACE_Z), PROBE_GRIP, "settle", YAW_L),
        Waypoint(1.5, (hx, hy, PROBE_PLACE_Z), OPEN, "let go", YAW_L),
        Waypoint(1.0, (hx, hy, PROBE_HOVER_Z), OPEN, "up", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
