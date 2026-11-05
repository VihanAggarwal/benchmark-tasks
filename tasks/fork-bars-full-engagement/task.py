"""On the station table: a brown crate (a closed, empty plywood crate, 110 mm along x, 300 mm along y, 50 mm tall,
0.15 kg) resting square on three dark-grey floor bearers (fixed timbers 36 mm tall, x 0.270 to 0.380) 27 to 38 cm in
front of the arm bases; the two 40 mm gaps between the bearers, centred on y +-0.120, are its fork pockets. Two grey
fork bars (loose square aluminium tubes, 160 mm long, 24 x 24 mm, 60 g each) lie on the table along x, each in line
with a pocket, their tips at the pocket mouths. White lines on the table beside each bar's path mark where its rear
end is when the bar is fully in (100 mm), yellow lines two-thirds in (73 mm). Beyond the crate stands a fixed blue
rack: a platform 36 mm tall carrying three bearers in the same plan, their tops 72 mm up.

The crate must go onto the rack's bearers on the two bars alone: both bars fully into the pockets, the crate lifted
level on both together (never on a bar less than two-thirds in, never tilted more than 5 degrees, its bottom never
above 9.5 cm), set down on the rack's bearers, the bars slid out, put down and let go; no gripper may touch the
crate. It is wider than a gripper opens on every side.

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
import offline_ride  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

# Reads both arms before every control step. False when yam_arena's embodiment is not loaded (offline: the checker
# feeds the arms instead).
_IN_SIMULATOR = rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with the crate on the rack, both bars out and down, both grippers open and clear.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Move the brown crate from the floor bearers onto the blue rack using only the two grey fork bars: slide both "
    "bars into the crate's fork pockets until their rear ends reach the white lines, lift the crate level on both "
    "bars together, set it on the rack's bearers and slide the bars out. Never lift with a bar less than two-thirds "
    "in, tilt the crate more than 5 degrees, raise its bottom above 9.5 cm, drop it or touch it with a gripper. Then "
    "put the bars down and let go."
)


def _fixed(name: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, color) -> Box:
    """A fixed box spanning x0..x1, y0..y1 and z0..z1 above the table top."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2),
               size=(x1 - x0, y1 - y0, z1 - z0), color=color)


def _bearers(prefix: str, x0: float, x1: float, z0: float, color) -> list[Box]:
    """A centre bearer and two outer bearers along x, 36 mm tall from z0, leaving the two fork pockets."""
    c, o0, o1, h = L.CENTRE_BEARER_Y, L.OUTER_BEARER_Y0, L.OUTER_BEARER_Y1, L.BEARER_H
    return [
        _fixed(f"{prefix}_centre", x0, x1, -c, c, z0, z0 + h, color),
        _fixed(f"{prefix}_left", x0, x1, o0, o1, z0, z0 + h, color),
        _fixed(f"{prefix}_right", x0, x1, -o1, -o0, z0, z0 + h, color),
    ]


def _line(name: str, x: float, y: float, color) -> Box:
    return Box(name, pos=(x, y, L.TOP + L.LINE_SIZE[2] / 2), size=L.LINE_SIZE, color=color)


FLOOR = _bearers("floor_bearer", L.FLOOR_X0, L.FLOOR_X1, 0.0, L.BEARER_GREY)
RACK = [_fixed("rack_platform", L.RACK_X0, L.RACK_X1, -L.RACK_HALF_Y, L.RACK_HALF_Y, 0.0, L.RACK_PLATFORM_H,
               L.RACK_BLUE),
        *_bearers("rack_bearer", L.RACK_BEARER_X0, L.RACK_BEARER_X1, L.RACK_PLATFORM_H, L.RACK_BEARER_BLUE)]
LINES = [
    _line("line_full_left", L.FULL_LINE_X, +L.LINE_Y, L.WHITE),
    _line("line_full_right", L.FULL_LINE_X, -L.LINE_Y, L.WHITE),
    _line("line_two_thirds_left", L.TWO_THIRDS_LINE_X, +L.LINE_Y, L.YELLOW),
    _line("line_two_thirds_right", L.TWO_THIRDS_LINE_X, -L.LINE_Y, L.YELLOW),
]
BAR_LEFT = Box("fork_bar_left", pos=(L.BAR_X, +L.POCKET_Y, L.TOP + L.BAR_HALF_H), size=L.BAR_SIZE,
               mass=L.BAR_MASS, color=L.BAR_GREY)
BAR_RIGHT = Box("fork_bar_right", pos=(L.BAR_X, -L.POCKET_Y, L.TOP + L.BAR_HALF_H), size=L.BAR_SIZE,
                mass=L.BAR_MASS, color=L.BAR_GREY)
CRATE = Box("crate", pos=(L.CRATE_X, L.CRATE_Y, L.TOP + L.FLOOR_TOP + L.CRATE_HALF[2]), size=L.CRATE_SIZE,
            mass=L.CRATE_MASS, color=L.CRATE_BROWN)
OBJECTS = [*FLOOR, *RACK, *LINES, BAR_LEFT, BAR_RIGHT, CRATE]  # the bars before the crate (offline_ride below)

# Offline checker only (tools/offline/check_task.py): it carries the bars but not the crate resting on them, so
# the crate's offline state comes from a small ride model. The hooks go on the object specs only when no simulator
# is loaded; in the simulator the specs stay exactly as yam_arena made them.
_RIDE = offline_ride.CrateRide((BAR_LEFT.name, BAR_RIGHT.name))
if not _IN_SIMULATOR:
    _RIDE.attach(CRATE, (BAR_LEFT, BAR_RIGHT))

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    _RIDE.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _bars(objects) -> tuple:
    return objects["fork_bar_left"], objects["fork_bar_right"]


def success(objects) -> bool:
    """Crate on the rack's bearers, every lift made with both bars fully in, both bars out from under it, both
    grippers open and 30 mm clear, no rule ever broken; held still 1 s. This runs first every control step, so it
    also feeds the tracker (with the arms rigstate read for this step)."""
    crate, bars = objects["crate"], _bars(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(crate, bars, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "crate_pos": crate.pos, "crate_quat": crate.quat_xyzw,
           "bar_left_pos": bars[0].pos, "bar_right_pos": bars[1].pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, the crate was lifted with a bar less than 73 mm into its pocket, tilted more than 5 deg,
    had its lowest point more than 95 mm up, fell, left the table, or a gripper's fingers touched it. Else 1.0 on
    the rack's bearers, every lift with both bars 95 mm or more in, both bars out from under it, both grippers open
    and clear; 0.75 on the rack otherwise; 0.5 lifted 20 mm on both bars and carried 50 mm toward the rack at some
    point; 0.25 both bars 95 mm or more in at the same time at some point; 0 otherwise."""
    return _TRACKER.result(objects["crate"], _bars(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.BAR_SIZE[1] - 0.008) / 2  # across the bar's 24 mm width
# Yaw 0 for both arms: the fingers close along y, across the bar; the gripper's 65 mm-wide body runs along x, so at
# full insertion it stays 12.5 mm short of the crate's front face. Both arms run the same plan, mirrored in y.
YAW = 0.0
GRIP_X0 = L.BAR_X - L.BAR_HALF_LEN + L.BAR_GRIP_FROM_REAR  # x 0.125: 15 mm from the bar's rear end
INSERT = L.FULL_LINE_X - (L.BAR_X - L.BAR_HALF_LEN)  # 0.100: the rear end from x 0.110 to the white line
GRIP_X1 = GRIP_X0 + INSERT  # x 0.225: the tip 100 mm in, the TCP 45 mm in front of the crate's face
BAR_TCP_Z = L.TOP + 0.018  # pad tips 5 mm above the table, on the bar's sides (19 mm of its 24 mm held)
APPROACH_Z = L.TOP + 0.040  # each final descent starts here and takes 1.5 s
HOVER = L.TOP + 0.100
# The bar's centre is 6 mm below the TCP, its top 6 mm above it. The crate's underside (36 mm) meets the bars' tops
# at TCP 0.030. The arms sag under the load (joints 1-3: 80 N m/rad, no gravity compensation for a payload): with a
# bar and half the crate on each arm, about 3-4.5 mm down and 4 mm back, and the gripper pitches about 1.2 deg, so
# the crate's far edge is about 8 mm lower than commanded (J K^-1 J^T, yamkin). Lifted to TCP 0.081 the crate's
# underside is commanded at 0.087: about 0.079 at its far edge in the simulator, 7 mm over the rack bearers' 0.072
# tops as it passes over them, and 8 mm under the 0.095 limit if the arms do not sag at all.
LIFT_Z = L.TOP + 0.081
# Carried 142 mm: the crate's centre from x 0.325 to 0.467 commanded, about 0.463 after the 4 mm sag back: its
# footprint x 0.408..0.518 on the rack bearers (0.410..0.520), 8 mm or more inside the 0.400..0.530 window for any
# sag from 0 to 8 mm.
CARRY = 0.142
GRIP_X2 = GRIP_X1 + CARRY  # x 0.367
SET_Z = L.TOP + 0.066  # crate's underside on the rack bearers' tops (0.072) with no sag: it is set down by here
# Bars lowered clear of the crate, into the 36 mm between its underside (0.072) and the platform (0.036): commanded
# 2 mm above the middle for the unloaded sag (1.9 mm down, 0.5 deg), so about 6 mm under the crate and 5 mm over
# the platform at the tips in the simulator (4 and 8 mm with no sag at all).
CLEAR_Z = L.TOP + 0.062
WITHDRAW_X = GRIP_X2 - 0.112  # x 0.255: the bars' tips at 0.400, out from under the crate (its face at 0.408+)
PUT_X = 0.140  # the bars come back to lie in line with the floor pockets, rear ends at x 0.125
PUT_Z = L.TOP + 0.020  # let go with the bar 2 mm above the table


def _arm(y: float) -> list[Waypoint]:
    """One arm's plan; the other arm runs the same at -y, on the same clock."""
    return [
        Waypoint(3.0, (GRIP_X0, y, HOVER), OPEN, "over the bar's rear end", YAW),
        Waypoint(2.0, (GRIP_X0, y, APPROACH_Z), OPEN, "down to just above it", YAW),
        Waypoint(1.5, (GRIP_X0, y, BAR_TCP_Z), OPEN, "slowly down beside it", YAW),
        Waypoint(0.5, (GRIP_X0, y, BAR_TCP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (GRIP_X0, y, BAR_TCP_Z), GRIP, "grip the bar", YAW),
        Waypoint(0.5, (GRIP_X0, y, BAR_TCP_Z), GRIP, "settle the grip", YAW),
        Waypoint(4.0, (GRIP_X1, y, BAR_TCP_Z), GRIP, "slide it into the pocket to the white line", YAW),
        Waypoint(0.5, (GRIP_X1, y, BAR_TCP_Z), GRIP, "settle", YAW),
        Waypoint(3.0, (GRIP_X1, y, LIFT_Z), GRIP, "lift the crate level on both bars", YAW),
        Waypoint(3.0, (GRIP_X2, y, LIFT_Z), GRIP, "carry it over the rack", YAW),
        Waypoint(2.0, (GRIP_X2, y, SET_Z), GRIP, "set it down on the rack's bearers", YAW),
        Waypoint(1.0, (GRIP_X2, y, CLEAR_Z), GRIP, "lower the bars clear of it", YAW),
        Waypoint(0.5, (GRIP_X2, y, CLEAR_Z), GRIP, "settle", YAW),
        Waypoint(3.0, (WITHDRAW_X, y, CLEAR_Z), GRIP, "slide the bar out from under it", YAW),
        Waypoint(2.0, (PUT_X, y, CLEAR_Z), GRIP, "bring it back over the floor pocket's line", YAW),
        Waypoint(2.0, (PUT_X, y, PUT_Z), GRIP, "put it down on the table", YAW),
        Waypoint(0.5, (PUT_X, y, PUT_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (PUT_X, y, PUT_Z), OPEN, "let go", YAW),
        Waypoint(2.0, (PUT_X, y, HOVER), OPEN, "up and clear", YAW),
    ]


def demo():
    """Both arms on one clock, the same plan mirrored in y. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm(+L.POCKET_Y), "right": _arm(-L.POCKET_Y)}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
