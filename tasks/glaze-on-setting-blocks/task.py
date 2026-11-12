"""On the station table: the bottom of a window frame (fixed, aluminium grey) on the midline 30 cm in front of the
arm bases: a sill (60 mm along x, 260 mm along y, 10 mm tall) carrying two glazing stops (8 mm thick, 40 mm tall)
that leave a 40 mm wide channel along y, closed at its ends by two short jambs (60 mm tall), so the channel is
240 mm long. Two white quarter-point marks (24 x 30 mm) lie on the sill in the channel at y -5 cm and +5 cm. Two
loose black rubber setting blocks (24 x 30 x 8 mm) lie side by side on a dark grey tray 24 cm right of the
midline; a loose glass unit (24 mm thick, 200 mm long, 80 mm tall, 0.20 kg) stands on its long edge on the table
29 cm left of the midline, its faces normal to x. Rule: the glass must be left standing on both setting blocks on
its own (both grippers let go) and must never touch the sill.

The scorer reads the arms (TCP, pad-tip point, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset; it changes nothing the embodiment does. Unreadable arms never count as let go.

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
# The 1.0 conditions already need the unit and both blocks let go (both grippers' finger zones 30 mm or more from
# each) on every step of the last 1 s and still over it, so a held or hovering unit can never start this hold.
SUCCESS_HOLD_S = 1.0

# The goal only: the order of work and how to bring the glass in are for the robot to work out.
INSTRUCTION = (
    "Stand the glass unit in the frame on the two black setting blocks, set on the white marks, and let go. "
    "The glass must never touch the sill."
)

_sill_z = L.TOP + L.SILL_SIZE[2] / 2
SILL = Box("sill", pos=(L.FRAME_X, L.FRAME_Y, _sill_z), size=L.SILL_SIZE, color=L.ALUMINIUM)
STOPS = [
    Box(name, pos=(x, L.FRAME_Y, L.SILL_TOP + L.STOP_SIZE[2] / 2), size=L.STOP_SIZE, color=L.ALUMINIUM)
    for name, x in (("stop_near", L.STOP_XS[0]), ("stop_far", L.STOP_XS[1]))
]
JAMBS = [
    Box(name, pos=(L.FRAME_X, y, L.SILL_TOP + L.JAMB_SIZE[2] / 2), size=L.JAMB_SIZE, color=L.ALUMINIUM)
    for name, y in (("jamb_right", L.JAMB_YS[0]), ("jamb_left", L.JAMB_YS[1]))
]
MARKS = [
    Box(name, pos=(L.FRAME_X, y, L.SILL_TOP + L.MARK_SIZE[2] / 2), size=L.MARK_SIZE, color=L.WHITE)
    for name, y in (("mark_right", L.MARK_YS[0]), ("mark_left", L.MARK_YS[1]))
]
TRAY = Box("block_tray", pos=(L.TRAY_X, L.TRAY_Y, L.TOP + L.TRAY_SIZE[2] / 2), size=L.TRAY_SIZE, color=L.TRAY_GREY)
BLOCKS = [
    Box(name, pos=(x, y, L.TRAY_TOP + L.BLOCK_SIZE[2] / 2), size=L.BLOCK_SIZE, mass=L.BLOCK_MASS, color=L.RUBBER)
    for name, (x, y) in zip(("block_1", "block_2"), L.BLOCK_STARTS)
]
UNIT = Box(
    "glass_unit",
    pos=(L.UNIT_X, L.UNIT_Y, L.TOP + L.UNIT_SIZE[2] / 2),
    size=L.UNIT_SIZE,
    mass=L.UNIT_MASS,
    color=L.GLASS,
)
OBJECTS = [SILL, *STOPS, *JAMBS, *MARKS, TRAY, *BLOCKS, UNIT]

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
    """Both blocks set on their marks, the unit standing on both of them in the channel, centred, never having
    touched the sill; the unit and both blocks let go (finger zones 30 mm or more from each on every step of the last
    1 s; unreadable arms are not let go) and still over that 1 s; held 1 s. This runs first every control step, so it
    also feeds the tracker (with the arms rigstate read for this step)."""
    unit, b1, b2 = objects["glass_unit"], objects["block_1"], objects["block_2"]
    arms = rigstate.arms()
    result = _TRACKER.update(unit, b1, b2, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "unit_pos": unit.pos, "unit_quat": unit.quat_xyzw, "unit_vel": unit.lin_vel,
           "block_1_pos": b1.pos, "block_2_pos": b2.pos}
    if rigstate.readable(arms):
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   openings=[arms["left"].opening, arms["right"].opening])
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the unit ever touched the sill, or ends lying down or off the table. Else 1.0 both blocks set within
    8 mm of their marks and the unit standing resting on both, centred within 8 mm; 0.75 the same with the blocks
    within 20 mm or the unit within 15 mm; 0.5 the unit standing in the channel resting on at least one block; 0.25
    both blocks set; 0 otherwise. 'Set' and 'resting' need the object let go (finger zones 30 mm or more from it on
    every step of the last 1 s; unreadable arms are not let go), still over that 1 s, and its support within 1 mm."""
    return _TRACKER.result(objects["glass_unit"], objects["block_1"], objects["block_2"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PAD = 0.013  # the pad tips are 13 mm below the TCP

# Right arm, yaw 0 (fingers close along y, across a block's 30 mm, so the pads sit along the channel; the pads'
# 27 mm width lies across the 40 mm channel).
BLOCK_GRIP = -(L.BLOCK_SIZE[1] - 0.008) / 2  # 22 mm commanded
BLOCK_LET_GO = -0.024  # 48 mm: 9 mm clear of the block on each side, well inside the channel's 240 mm length
BLOCK_TCP_UP = 0.0025 + PAD  # the TCP this far above a block's bottom: pad tips 2.5 mm above it
B_PICK_Z = L.TRAY_TOP + BLOCK_TCP_UP
B_PLACE_Z = L.MARK_TOP + 0.0015 + BLOCK_TCP_UP  # let go with the block's bottom 1.5 mm above its mark
B_CARRY_Z = L.STOP_TOP + 0.015 + BLOCK_TCP_UP  # block 15 mm above the stops (pad tips 17.5 mm above them)
HOVER_R = L.TOP + 0.10  # pad tips 17 mm above the jambs' tops
LANE_X = 0.25  # both arms come to the channel across the near stop from here: well clear of the jambs (x >= 0.28)
PARK_R = (0.22, -0.26, L.TOP + 0.12)
# Where each block is set, from its mark's centre (x, y): (right mark, left mark). Zero offline. On the GPU, read
# block_mark_mm in TASK_TRACE after each release: if a block lands consistently off its mark (the pads are not centred
# on the commanded point; another team measured up to about 8 mm), trim here. Keep x within +-3 mm: the pads have
# 6.5 mm to each stop.
BLOCK_TRIM = ((0.0, 0.0), (0.0, 0.0))

# Left arm, yaw -pi/2 (fingers close along x, across the unit's 24 mm thickness).
YAW_L = -math.pi / 2
UNIT_GRIP = -(L.UNIT_SIZE[0] - 0.008) / 2  # 16 mm commanded
UNIT_LET_GO = -0.024  # 48 mm: 12 mm clear of each face; the pad tips 22 mm above the stops (about 16 with the sag)
UNIT_TCP_UP = L.UNIT_SIZE[2] - 0.016  # grip the top edge: the TCP 16 mm below it, pad tips 29 mm below it
U_PICK_Z = L.TOP + UNIT_TCP_UP
# Payload sag: the arm's controller does not compensate the 0.20 kg unit's weight. The static estimate (J K^-1 J^T of
# the weight at the TCP; joints 80/80/80/10/10/10 N m/rad; the unit hangs straight below the TCP, so no wrist moment)
# is 5.6 mm down, 1.3 mm sideways, 0.9 deg of tilt, over the channel. So the carry clears the stops by 18 mm as
# commanded (about 12 mm in the simulator), and the unit is let go 2 mm above the blocks as commanded: with the sag it
# already rests on them (no drop; the blocks take its weight as the pads open).
U_CARRY_Z = L.STOP_TOP + 0.018 + UNIT_TCP_UP
U_PLACE_Z = L.SET_BLOCK_TOP + 0.002 + UNIT_TCP_UP
U_ENTRY_Z = U_PLACE_Z + 0.020  # the last 20 mm are lowered slowly
HOVER_L = L.TOP + 0.14
# After letting go the hand goes straight up and then back over the unit's (now empty) starting spot, at the height it
# first came over the unit: its finger zone (rigstate.py, 35 mm round the pads' axis at the 48 mm let-go opening) ends
# about 12 cm (117 mm offline) from the glass, well past the scorer's 30 mm 'let go'. Straight up alone (pad tips 18 mm over the
# glass) would not count: the zone reaches below the pad tips.
RETREAT_L = (L.UNIT_X, L.UNIT_Y, HOVER_L)
# Where the unit is set down, from the channel's centre (x, y). The sideways sag moves it about (-1.3, +1.3) mm, well
# inside the channel's 8 mm play each side and the 8 mm centring limit, so no trim. On the GPU, read unit_pos in
# TASK_TRACE at the end: if the unit hangs off the TCP's line (a pad that leads on closing shoves it), trim here.
PLACE_TRIM = (0.0, 0.0)


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _place_block(start, mark: int, first: bool, lane: bool) -> list[Waypoint]:
    """Pick one block off the tray, carry it across the near stop and set it on a mark (0 right, 1 left), let go, go
    up. With ``lane`` it first goes beside the frame level with the mark (the right mark: a straight line from the
    tray would take the fingers over the right jamb's corner); without, straight over the mark (2.6 cm clear of that
    jamb). The pads (27 mm wide across the 40 mm channel) go down past the stops over 2 s, and every grip and release
    takes 1.5 s (the CLI's 0.05-per-step gripper limiter; slow, even releases do not kick a 7 g block)."""
    sx, sy = start
    mx, my = L.FRAME_X + BLOCK_TRIM[mark][0], L.MARK_YS[mark] + BLOCK_TRIM[mark][1]
    to_mark = [Waypoint(1.5, (LANE_X, my, B_CARRY_Z), BLOCK_GRIP, "beside the frame, level with the mark"),
               Waypoint(0.8, (mx, my, B_CARRY_Z), BLOCK_GRIP, "across the near stop, over the mark")]
    if not lane:
        to_mark = [Waypoint(2.0, (mx, my, B_CARRY_Z), BLOCK_GRIP, "across the near stop, over the mark")]
    return [
        Waypoint(2.0 if first else 1.8, (sx, sy, HOVER_R), OPEN, "over a setting block"),
        Waypoint(0.8, (sx, sy, B_PICK_Z + 0.030), OPEN, "down to just above it"),
        Waypoint(1.2, (sx, sy, B_PICK_Z), OPEN, "slowly down beside its 30 mm sides"),
        Waypoint(0.5, (sx, sy, B_PICK_Z), OPEN, "settle"),
        Waypoint(1.5, (sx, sy, B_PICK_Z), BLOCK_GRIP, "grip the block"),
        Waypoint(1.0, (sx, sy, B_CARRY_Z), BLOCK_GRIP, "lift"),
        *to_mark,
        Waypoint(2.0, (mx, my, B_PLACE_Z), BLOCK_GRIP, "slowly down into the channel onto the mark"),
        Waypoint(0.5, (mx, my, B_PLACE_Z), BLOCK_GRIP, "settle"),
        Waypoint(1.5, (mx, my, B_PLACE_Z), BLOCK_LET_GO, "let go of the block"),
        Waypoint(1.0, (mx, my, HOVER_R), BLOCK_LET_GO, "straight up out of the channel"),
    ]


def _right() -> list[Waypoint]:
    """Right arm: set the block nearer the arms on the right mark, then the other on the left mark (while the left
    arm holds the glass back out of its way), then move out of the left arm's way."""
    return [
        *_place_block(L.BLOCK_STARTS[0], 0, first=True, lane=True),
        *_place_block(L.BLOCK_STARTS[1], 1, first=False, lane=False),
        Waypoint(1.5, PARK_R, OPEN, "out of the way, on its own side"),
    ]


def _left_pick() -> list[Waypoint]:
    ux, uy = L.UNIT_X, L.UNIT_Y
    return [
        Waypoint(2.0, (ux, uy, HOVER_L), OPEN, "over the glass unit", YAW_L),
        Waypoint(1.0, (ux, uy, U_PICK_Z + 0.030), OPEN, "down to just above its top edge", YAW_L),
        Waypoint(1.5, (ux, uy, U_PICK_Z), OPEN, "slowly down beside its faces", YAW_L),
        Waypoint(0.5, (ux, uy, U_PICK_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (ux, uy, U_PICK_Z), UNIT_GRIP, "grip its top edge", YAW_L),
        Waypoint(0.5, (ux, uy, U_PICK_Z), UNIT_GRIP, "settle the grip", YAW_L),
        Waypoint(1.5, (ux, uy, U_CARRY_Z), UNIT_GRIP, "lift it straight up", YAW_L),
    ]


# While the right arm sets the left-hand block its wrist (link4) leans toward the left arm: the left arm holds the
# unit 2 cm further back and 2 cm further out meanwhile (arm-arm clearance 8.8 cm in the offline check). It starts
# its carry once the right arm is parked on its own side.
WAIT_L = (L.UNIT_X - 0.020, L.UNIT_Y + 0.020)
_RIGHT_CLEAR_AT = _span(_right())


def _left() -> list[Waypoint]:
    """Left arm: grip the unit's top edge, lift it, wait for the blocks, bring it beside the frame and across the
    near stop over the channel, lower it slowly onto the blocks, let go, go up and back clear of the glass."""
    fx, fy = L.FRAME_X + PLACE_TRIM[0], L.FRAME_Y + PLACE_TRIM[1]
    pick = _left_pick() + [Waypoint(1.0, (*WAIT_L, U_CARRY_Z), UNIT_GRIP, "back a little, out of the way", YAW_L)]
    wait = _RIGHT_CLEAR_AT - _span(pick)
    return pick + [
        Waypoint(wait, (*WAIT_L, U_CARRY_Z), UNIT_GRIP, "hold it up while the blocks go in", YAW_L),
        Waypoint(2.0, (LANE_X, fy, U_CARRY_Z), UNIT_GRIP, "beside the frame, level with the channel", YAW_L),
        Waypoint(1.2, (fx, fy, U_CARRY_Z), UNIT_GRIP, "across the near stop, over the channel", YAW_L),
        Waypoint(0.5, (fx, fy, U_CARRY_Z), UNIT_GRIP, "settle", YAW_L),
        Waypoint(1.5, (fx, fy, U_ENTRY_Z), UNIT_GRIP, "down into the channel to 2 cm above the blocks", YAW_L),
        Waypoint(0.8, (fx, fy, U_ENTRY_Z), UNIT_GRIP, "settle over the blocks", YAW_L),
        Waypoint(2.5, (fx, fy, U_PLACE_Z), UNIT_GRIP, "lower the last 2 cm slowly", YAW_L),
        Waypoint(0.5, (fx, fy, U_PLACE_Z), UNIT_GRIP, "settle on the blocks", YAW_L),
        Waypoint(1.5, (fx, fy, U_PLACE_Z), UNIT_LET_GO, "let go", YAW_L),
        Waypoint(1.0, (fx, fy, HOVER_L - 0.01), UNIT_LET_GO, "straight up", YAW_L),
        Waypoint(2.0, RETREAT_L, UNIT_LET_GO, "back over its starting spot, clear of the glass", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
