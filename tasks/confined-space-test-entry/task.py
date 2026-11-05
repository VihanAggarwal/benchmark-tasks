"""On the station table: a closed grey steel tank (160 x 480 mm, 66 mm tall, fixed) 28-44 cm in front of the arm
bases, its deck pierced by two 120 x 50 mm manways at y +-18 cm, each closed by a loose dark grey cover; a grey
stand beside the tank in front of each arm for its cover; a blue gas probe (a 16 x 100 mm rod) standing in a fixed
black holder in front of the left arm; a white depth gauge with red, yellow and green bands just in front of the
tank, in line with the left manway; an orange tool lying on the tank floor straight under the right manway; and a
white pad for the tool in front of the right arm. Confined-space entry: both covers off first, then the air tested
at three depths through the left manway, top first, holding the probe still 2 s at each; no gripper in the tank
before that, and the probe held in the tank whenever a gripper is in it; then the tool out onto the pad, the probe
back in its holder and both covers back on.

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
from scoring import Tracker, arms_valid  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with both grippers open and clear of everything (the demo ends with both well above the covers).
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Take the orange tool out of the grey tank and put it on the white pad. First lift both manway covers off onto "
    "their grey stands. Then lower the blue gas probe through the left manway and hold its tip still for 2 s at each "
    "of the three depths marked on the gauge, top first. No gripper may go below the tank's top before that, and "
    "the probe must stay held inside the tank whenever a gripper is in it. Finish with the probe in its holder, both "
    "covers back on and both grippers open."
)


def _box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A fixed box from its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2),
               size=(x1 - x0, y1 - y0, z1 - z0), color=color, **kw)


def _tank() -> list[Box]:
    """Floor plate, four walls and a deck with two manway openings, all fixed."""
    x0, x1, y0, y1, t = L.TANK_X0, L.TANK_X1, L.TANK_Y0, L.TANK_Y1, L.WALL_T
    mx0, mx1 = L.MANWAY_X - L.MANWAY_HALF[0], L.MANWAY_X + L.MANWAY_HALF[0]
    ly0, ly1 = L.MANWAY_Y["left"] - L.MANWAY_HALF[1], L.MANWAY_Y["left"] + L.MANWAY_HALF[1]
    ry0, ry1 = L.MANWAY_Y["right"] - L.MANWAY_HALF[1], L.MANWAY_Y["right"] + L.MANWAY_HALF[1]
    w0, w1, d0, d1 = L.FLOOR_H, L.WALL_TOP_H, L.WALL_TOP_H, L.DECK_H
    s = L.STEEL
    return [
        _box("tank_floor", x0, x1, y0, y1, 0.0, L.FLOOR_H, s),
        _box("tank_wall_front", x1 - t, x1, y0, y1, w0, w1, s),
        _box("tank_wall_back", x0, x0 + t, y0, y1, w0, w1, s),
        _box("tank_wall_left", x0 + t, x1 - t, y1 - t, y1, w0, w1, s),
        _box("tank_wall_right", x0 + t, x1 - t, y0, y0 + t, w0, w1, s),
        _box("deck_back", x0, mx0, y0, y1, d0, d1, s),
        _box("deck_front", mx1, x1, y0, y1, d0, d1, s),
        _box("deck_middle", mx0, mx1, ry1, ly0, d0, d1, s),
        _box("deck_left_edge", mx0, mx1, ly1, y1, d0, d1, s),
        _box("deck_right_edge", mx0, mx1, y0, ry0, d0, d1, s),
    ]


def _holder() -> list[Box]:
    """Four fixed walls round the probe's 24 x 24 mm socket; the table is its floor."""
    x, y, h = L.HOLDER_X, L.HOLDER_Y, L.HOLDER_H
    a, w = L.HOLDER_INNER / 2, L.HOLDER_WALL
    c = L.HOLDER_BLACK
    return [
        _box("holder_front", x + a, x + a + w, y - a - w, y + a + w, 0.0, h, c),
        _box("holder_back", x - a - w, x - a, y - a - w, y + a + w, 0.0, h, c),
        _box("holder_left", x - a, x + a, y + a, y + a + w, 0.0, h, c),
        _box("holder_right", x - a, x + a, y - a - w, y - a, 0.0, h, c),
    ]


def _gauge() -> list[Box]:
    """The depth gauge: white post sections and 12 mm red, yellow and green bands stacked from the table up."""
    x, y = L.GAUGE_X, L.GAUGE_Y
    out, z = [], 0.0
    names = {"top": "gauge_red", "middle": "gauge_yellow", "bottom": "gauge_green"}
    for i, (band, lo, hi) in enumerate(reversed(L.BANDS)):  # bottom band first
        p = L.GAUGE_POST / 2
        out.append(_box(f"gauge_post_{i}", x - p, x + p, y - p, y + p, z, lo, L.WHITE))
        b = L.GAUGE_BAND / 2
        out.append(_box(names[band], x - b, x + b, y - b, y + b, lo, hi, L.BAND_COLOURS[band]))
        z = hi
    p = L.GAUGE_POST / 2
    out.append(_box("gauge_post_3", x - p, x + p, y - p, y + p, z, L.DECK_H, L.WHITE))
    return out


COVERS = [
    Box(f"cover_{side}", pos=(L.MANWAY_X, L.MANWAY_Y[side], L.DECK_Z + L.COVER_SIZE[2] / 2), size=L.COVER_SIZE,
        mass=L.COVER_MASS, color=L.COVER_GREY)
    for side in ("left", "right")
]
STANDS = [
    _box(f"stand_{side}", L.STAND_X - L.STAND_SIZE[0] / 2, L.STAND_X + L.STAND_SIZE[0] / 2,
         L.STAND_Y[side] - L.STAND_SIZE[1] / 2, L.STAND_Y[side] + L.STAND_SIZE[1] / 2, 0.0, L.STAND_SIZE[2],
         L.STAND_GREY)
    for side in ("left", "right")
]
PROBE = Cylinder("probe", pos=(L.HOLDER_X, L.HOLDER_Y, L.TOP + L.PROBE_LEN / 2), radius=L.PROBE_RADIUS,
                 height=L.PROBE_LEN, mass=L.PROBE_MASS, color=L.PROBE_BLUE)
TOOL = Box("tool", pos=(L.TOOL_X, L.TOOL_Y, L.FLOOR_Z + L.TOOL_SIZE[2] / 2), size=L.TOOL_SIZE, mass=L.TOOL_MASS,
           color=L.ORANGE)
PAD = _box("tool_pad", L.PAD_X - L.PAD_SIZE[0] / 2, L.PAD_X + L.PAD_SIZE[0] / 2, L.PAD_Y - L.PAD_SIZE[1] / 2,
           L.PAD_Y + L.PAD_SIZE[1] / 2, 0.0, L.PAD_SIZE[2], L.WHITE)
OBJECTS = [*_tank(), *COVERS, *STANDS, PROBE, *_holder(), *_gauge(), TOOL, PAD]

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


def _objs(objects):
    return objects["cover_left"], objects["cover_right"], objects["probe"], objects["tool"]


def success(objects) -> bool:
    """Tool flat on the pad, probe upright in its holder, both covers back on their manways, both grippers open and
    30 mm clear of everything, no rule broken at any time; held still 1 s. This runs first every control step, so it
    also feeds the tracker (with the arms rigstate read for this step)."""
    cl, cr, probe, tool = _objs(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(cl, cr, probe, tool, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "probe_pos": probe.pos, "tool_pos": tool.pos, "cover_left_pos": cl.pos,
           "cover_right_pos": cr.pos}
    if arms_valid(arms):
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:  # None, an arm missing or a non-finite value: the gripper rules were not applied this step
        row["rigstate_error"] = rigstate.error() or "arms incomplete or non-finite"
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, a gripper was in the tank before both covers were off and the three-depth test was
    complete, or while the probe was not held in the left manway's column; if the probe was let go with its tip in
    the tank; or if a cover, the probe or the tool fell, tipped over or left the table. Else 1.0 tool on the pad,
    probe in its holder, both covers back on, both grippers open and clear; 0.75 tool on the pad; 0.5 tool lifted
    out of the tank at some point; 0.25 test complete; 0 otherwise."""
    return _TRACKER.result(*_objs(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
COVER_GRIP = -(L.COVER_SIZE[1] - 0.008) / 2  # yaw 0: across the cover's 70 mm
PROBE_GRIP = -(2 * L.PROBE_RADIUS - 0.008) / 2  # yaw 0: across the probe's 16 mm
TOOL_GRIP = -(L.TOOL_SIZE[0] - 0.008) / 2  # yaw +pi/2 (right arm): across the tool's 60 mm length, along x
# The right gripper goes through the 120 x 50 mm manway with its fingers closing along x, 74 mm apart: the pads 7 mm
# clear of each end of the 60 mm tool and their outer faces at +-45 mm in x (15 mm inside the opening's ends; the
# fingers step out about 6.5 mm above the pads, so about 8 mm at the deck); across, the fingers are 27 mm wide at the
# pads and about 35 mm at the deck, in the 50 mm opening (about 7 mm each side).
TOOL_ENTRY = -0.037
YAW_TOOL = math.pi / 2


def _z(h: float) -> float:
    return L.TOP + h


HOVER = 0.130  # heights above the table top
RISE = 0.150  # where each arm ends: finger tips 51 mm over a cover, clear by the end rule's 30 mm
COVER_UP = 0.010  # each cover is gripped with the TCP 10 mm above its centre: pad tips 7 mm above its bottom
COVER_TCP = L.DECK_H + L.COVER_SIZE[2] / 2 + COVER_UP  # 0.086 on the deck: pad tips 7 mm above the deck
COVER_ON_STAND = L.STAND_SIZE[2] + 0.002 + L.COVER_SIZE[2] / 2 + COVER_UP  # let go 2 mm above the stand
COVER_FROM_STAND = L.STAND_SIZE[2] + L.COVER_SIZE[2] / 2 + COVER_UP
COVER_ON_DECK = L.DECK_H + 0.002 + L.COVER_SIZE[2] / 2 + COVER_UP  # let go 2 mm above the deck
TIP_BELOW_TCP = L.PROBE_LEN - L.PROBE_GRIP_DOWN  # 0.075: the probe's tip hangs this far below the TCP
PROBE_PICK = TIP_BELOW_TCP  # tip on the table, in the socket
PROBE_CARRY = 0.155  # tip 80 mm up: 14 mm over the deck while it is carried over the tank's back edge
PROBE_SET = TIP_BELOW_TCP + 0.002  # let go with the tip 2 mm above the table
PROBE_INTO_SOCKET = TIP_BELOW_TCP + 0.030  # tip 5 mm above the socket's 25 mm walls
DWELL_TIP = (0.050, 0.033, 0.016)  # tip heights for the three tests: 5 to 6 mm inside each 11 mm band
DWELL_HOLD_S = 4.5  # each test waypoint: 2.5 s more than the 2 s the scorer needs: the probe settles, and the
#                     scorer's speed is the tip's path over the last 0.5 s, so the 2 s start 0.5 s after it stops
TOOL_UP = 0.009  # the tool is gripped with the TCP 9 mm above its centre: pad tips 6 mm above its bottom
TOOL_TCP = L.FLOOR_H + L.TOOL_SIZE[2] / 2 + TOOL_UP  # 0.027: pad tips 6 mm above the tank floor, deck 39 mm up
TOOL_BELOW_TCP = L.TOOL_SIZE[2] / 2 + TOOL_UP  # 0.019: its bottom this far below the TCP
TOOL_CARRY = 0.110  # tool bottom 91 mm up: 25 mm over the deck
TOOL_SET = L.PAD_SIZE[2] + 0.002 + TOOL_BELOW_TCP  # let go 2 mm above the pad
INTO_TANK = 0.060  # the right gripper's descent through the manway slows from here (pad tips 47 mm up)
WAIT_MARGIN_S = 0.5  # each arm waits this much longer than the other arm's plan says it needs


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _cover_off(side: str) -> list[Waypoint]:
    """Lift a cover straight up off its manway and set it on its stand."""
    x, y, sy = L.MANWAY_X, L.MANWAY_Y[side], L.STAND_Y[side]
    return [
        Waypoint(2.5, (x, y, _z(HOVER)), OPEN, "over the cover"),
        Waypoint(1.0, (x, y, _z(COVER_TCP + 0.020)), OPEN, "down to 2 cm above the grip"),
        Waypoint(1.5, (x, y, _z(COVER_TCP)), OPEN, "slowly down beside its long sides"),
        Waypoint(0.5, (x, y, _z(COVER_TCP)), OPEN, "settle"),
        Waypoint(1.5, (x, y, _z(COVER_TCP)), COVER_GRIP, "grip the cover"),
        Waypoint(0.5, (x, y, _z(COVER_TCP)), COVER_GRIP, "settle the grip"),
        Waypoint(1.5, (x, y, _z(HOVER)), COVER_GRIP, "lift it straight up off the manway"),
        Waypoint(2.5, (x, sy, _z(HOVER)), COVER_GRIP, "carry it over its stand"),
        Waypoint(1.5, (x, sy, _z(COVER_ON_STAND + 0.020)), COVER_GRIP, "down to 2 cm above the stand"),
        Waypoint(1.5, (x, sy, _z(COVER_ON_STAND)), COVER_GRIP, "lower the last 2 cm slowly"),
        Waypoint(0.5, (x, sy, _z(COVER_ON_STAND)), COVER_GRIP, "settle"),
        Waypoint(1.5, (x, sy, _z(COVER_ON_STAND)), OPEN, "let go on the stand"),
        Waypoint(0.5, (x, sy, _z(COVER_ON_STAND)), OPEN, "settle"),
        Waypoint(1.0, (x, sy, _z(RISE)), OPEN, "up"),
    ]


def _cover_on(side: str) -> list[Waypoint]:
    """Take a cover from its stand and set it back on its manway; end well above it."""
    x, y, sy = L.MANWAY_X, L.MANWAY_Y[side], L.STAND_Y[side]
    return [
        Waypoint(2.5, (x, sy, _z(HOVER)), OPEN, "over the cover on its stand"),
        Waypoint(1.5, (x, sy, _z(COVER_FROM_STAND + 0.020)), OPEN, "down to 2 cm above the grip"),
        Waypoint(1.5, (x, sy, _z(COVER_FROM_STAND)), OPEN, "slowly down beside its long sides"),
        Waypoint(0.5, (x, sy, _z(COVER_FROM_STAND)), OPEN, "settle"),
        Waypoint(1.5, (x, sy, _z(COVER_FROM_STAND)), COVER_GRIP, "grip the cover"),
        Waypoint(0.5, (x, sy, _z(COVER_FROM_STAND)), COVER_GRIP, "settle the grip"),
        Waypoint(1.5, (x, sy, _z(HOVER)), COVER_GRIP, "lift it off the stand"),
        Waypoint(2.5, (x, y, _z(HOVER)), COVER_GRIP, "carry it over its manway"),
        Waypoint(1.0, (x, y, _z(COVER_ON_DECK + 0.020)), COVER_GRIP, "down to 2 cm above the deck"),
        Waypoint(0.5, (x, y, _z(COVER_ON_DECK + 0.020)), COVER_GRIP, "settle"),
        Waypoint(1.5, (x, y, _z(COVER_ON_DECK)), COVER_GRIP, "lower the last 2 cm slowly"),
        Waypoint(0.5, (x, y, _z(COVER_ON_DECK)), COVER_GRIP, "settle"),
        Waypoint(1.5, (x, y, _z(COVER_ON_DECK)), OPEN, "let go on the manway"),
        Waypoint(0.5, (x, y, _z(COVER_ON_DECK)), OPEN, "settle"),
        Waypoint(1.0, (x, y, _z(RISE)), OPEN, "up, clear of the cover"),
    ]


def _left_test() -> list[Waypoint]:
    """Left arm, up to the end of the air test: its cover off, the probe out of its holder and down through the
    left manway, held still 4.5 s at each depth, top first."""
    hx, hy = L.HOLDER_X, L.HOLDER_Y
    mx, my = L.MANWAY_X, L.MANWAY_Y["left"]
    t_top, t_mid, t_bot = (TIP_BELOW_TCP + h for h in DWELL_TIP)
    return _cover_off("left") + [
        Waypoint(2.5, (hx, hy, _z(HOVER)), OPEN, "over the probe"),
        Waypoint(1.0, (hx, hy, _z(PROBE_PICK + 0.020)), OPEN, "down to 2 cm above the grip"),
        Waypoint(1.5, (hx, hy, _z(PROBE_PICK)), OPEN, "slowly down beside it, 25 mm below its top"),
        Waypoint(0.5, (hx, hy, _z(PROBE_PICK)), OPEN, "settle"),
        Waypoint(1.5, (hx, hy, _z(PROBE_PICK)), PROBE_GRIP, "grip the probe"),
        Waypoint(0.5, (hx, hy, _z(PROBE_PICK)), PROBE_GRIP, "settle the grip"),
        Waypoint(2.0, (hx, hy, _z(PROBE_CARRY)), PROBE_GRIP, "lift it straight out of the holder"),
        Waypoint(3.0, (mx, my, _z(PROBE_CARRY)), PROBE_GRIP, "carry it over the left manway"),
        Waypoint(0.5, (mx, my, _z(PROBE_CARRY)), PROBE_GRIP, "settle"),
        Waypoint(2.0, (mx, my, _z(t_top)), PROBE_GRIP, "lower the tip to the top depth"),
        Waypoint(DWELL_HOLD_S, (mx, my, _z(t_top)), PROBE_GRIP, "test the top depth"),
        Waypoint(1.5, (mx, my, _z(t_mid)), PROBE_GRIP, "lower the tip to the middle depth"),
        Waypoint(DWELL_HOLD_S, (mx, my, _z(t_mid)), PROBE_GRIP, "test the middle depth"),
        Waypoint(1.5, (mx, my, _z(t_bot)), PROBE_GRIP, "lower the tip to the bottom depth"),
        Waypoint(DWELL_HOLD_S, (mx, my, _z(t_bot)), PROBE_GRIP, "test the bottom depth"),
    ]


TEST_DONE_S = _span(_left_test())  # the last test waypoint has ended (the scorer completes it about 1 s earlier)


def _right_tool() -> list[Waypoint]:
    """Right arm, up to the tool being out of the tank: its cover off; wait over the right manway, fingers 74 mm
    apart along x, until the test is complete; down through the manway, grip the tool, lift it out."""
    x, y = L.MANWAY_X, L.MANWAY_Y["right"]
    head = _cover_off("right") + [
        Waypoint(2.5, (x, y, _z(HOVER)), TOOL_ENTRY, "over the right manway, fingers along x", YAW_TOOL),
    ]
    wait = TEST_DONE_S + WAIT_MARGIN_S - _span(head)
    return head + [
        Waypoint(wait, (x, y, _z(HOVER)), TOOL_ENTRY, "wait until the air test is complete", YAW_TOOL),
        Waypoint(1.5, (x, y, _z(INTO_TANK)), TOOL_ENTRY, "down into the manway", YAW_TOOL),
        Waypoint(2.0, (x, y, _z(TOOL_TCP)), TOOL_ENTRY, "slowly down round the tool", YAW_TOOL),
        Waypoint(0.5, (x, y, _z(TOOL_TCP)), TOOL_ENTRY, "settle", YAW_TOOL),
        Waypoint(1.5, (x, y, _z(TOOL_TCP)), TOOL_GRIP, "grip the tool by its ends", YAW_TOOL),
        Waypoint(0.5, (x, y, _z(TOOL_TCP)), TOOL_GRIP, "settle the grip", YAW_TOOL),
        Waypoint(2.5, (x, y, _z(TOOL_CARRY)), TOOL_GRIP, "lift it straight out of the manway", YAW_TOOL),
    ]


TOOL_OUT_S = _span(_right_tool())  # the right gripper is up out of the tank with the tool


def _right() -> list[Waypoint]:
    """Right arm: tool out onto the pad, then its cover back on."""
    px, py = L.PAD_X, L.PAD_Y
    return _right_tool() + [
        Waypoint(3.0, (px, py, _z(TOOL_CARRY)), TOOL_GRIP, "carry it over the pad", YAW_TOOL),
        Waypoint(1.5, (px, py, _z(TOOL_SET + 0.020)), TOOL_GRIP, "down to 2 cm above the pad", YAW_TOOL),
        Waypoint(1.5, (px, py, _z(TOOL_SET)), TOOL_GRIP, "lower the last 2 cm slowly", YAW_TOOL),
        Waypoint(0.5, (px, py, _z(TOOL_SET)), TOOL_GRIP, "settle", YAW_TOOL),
        Waypoint(1.5, (px, py, _z(TOOL_SET)), OPEN, "let go on the pad", YAW_TOOL),
        Waypoint(0.5, (px, py, _z(TOOL_SET)), OPEN, "settle", YAW_TOOL),
        Waypoint(1.0, (px, py, _z(HOVER)), OPEN, "up", YAW_TOOL),
    ] + _cover_on("right")


def _left() -> list[Waypoint]:
    """Left arm: after the test, keep the probe still at the bottom depth until the right gripper is out of the
    tank, then lift it out, stand it back in its holder, and put the left cover back on."""
    hx, hy = L.HOLDER_X, L.HOLDER_Y
    mx, my = L.MANWAY_X, L.MANWAY_Y["left"]
    test = _left_test()
    t_bot = TIP_BELOW_TCP + DWELL_TIP[2]
    hold = TOOL_OUT_S + WAIT_MARGIN_S - _span(test)
    return test + [
        Waypoint(hold, (mx, my, _z(t_bot)), PROBE_GRIP, "keep monitoring while the right gripper is in"),
        Waypoint(2.0, (mx, my, _z(PROBE_CARRY)), PROBE_GRIP, "lift the probe out of the tank"),
        Waypoint(3.0, (hx, hy, _z(PROBE_CARRY)), PROBE_GRIP, "carry it over its holder"),
        Waypoint(0.5, (hx, hy, _z(PROBE_CARRY)), PROBE_GRIP, "settle"),
        Waypoint(1.5, (hx, hy, _z(PROBE_INTO_SOCKET)), PROBE_GRIP, "down to 5 mm above the socket"),
        Waypoint(0.5, (hx, hy, _z(PROBE_INTO_SOCKET)), PROBE_GRIP, "settle"),
        Waypoint(2.0, (hx, hy, _z(PROBE_SET)), PROBE_GRIP, "lower it slowly into the socket"),
        Waypoint(0.5, (hx, hy, _z(PROBE_SET)), PROBE_GRIP, "settle"),
        Waypoint(1.5, (hx, hy, _z(PROBE_SET)), OPEN, "let go"),
        Waypoint(0.5, (hx, hy, _z(PROBE_SET)), OPEN, "settle"),
        Waypoint(1.5, (hx, hy, _z(RISE)), OPEN, "up, clear of the probe"),
    ] + _cover_on("left")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
