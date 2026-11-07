"""On the station table: two white layout pads (the blocker's lens tables, 120 x 120 mm, fixed) 24 cm either side of
the midline, 30 cm in front of the arm bases, each holding a fixed uncut spectacle lens (a pale blue disc 70 mm
across) inked with a lensmeter's marks: a black dot on the optical centre, a black 60 mm line through it (the 180
line), turned 25 deg (left lens) or 30 deg (right lens) on its pad, and a small black square at its nasal end, toward
the midline. Two loose blue edger blocks (18 x 24 x 16 mm) stand on the table in front of the arms. Task: block both
lenses for edging at the frame's boxing centre, 8 mm temporal of each optical centre along its line, each block's
long side along the line, then let go.

The scorer reads the arms (TCP, gripper opening) through rigstate.py, which wraps the embodiment's step and reset from
this file's import; it changes nothing the embodiment does.

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
# The trial ends once both lenses have been blocked on target for 1 s. Success needs both blocks let go (no closed
# gripper within 40 mm of either), so the hold can only begin, and end, with the grippers open.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Block both lenses for edging at the frame's boxing centre. For this job (eye size 54 mm, bridge 20 mm, PD 58 mm) "
    "that is (54 + 20 - 58) / 2 = 8 mm temporal of the optical centre. On each lens, set one blue block flat with its "
    "centre 8 mm from the centre of the black dot (the optical centre) along the black ink line, on the side away "
    "from the small black square (the nasal mark), with the block's long side along the line. Then let go of both "
    "blocks."
)


def _lens_parts(side: str) -> list:
    """One layout pad, its lens and the lens's three marks (all fixed)."""
    ox, oy = L.OC[side]
    yaw = math.radians(L.line_heading_deg(side))
    nx, ny = L.NASAL_XY[side]
    return [
        Box(f"pad_{side}", pos=(ox, oy, L.TOP + L.PAD_SIZE[2] / 2), size=L.PAD_SIZE, color=L.PAD_WHITE),
        Cylinder(f"lens_{side}", pos=(ox, oy, L.TOP + L.LENS_Z0 + L.LENS_T / 2), radius=L.LENS_R, height=L.LENS_T,
                 color=L.LENS_BLUE),
        Box(f"line_{side}", pos=(ox, oy, L.TOP + L.MARK_H), size=L.LINE_SIZE, yaw=yaw, color=L.INK),
        Cylinder(f"oc_dot_{side}", pos=(ox, oy, L.TOP + L.MARK_H), radius=L.DOT_R, height=L.MARK_T, color=L.INK),
        Box(f"nasal_{side}", pos=(nx, ny, L.TOP + L.MARK_H), size=L.NASAL_SIZE, yaw=yaw, color=L.INK),
    ]


BLOCK_OBJECTS = [
    Box(name, pos=(*L.BLOCK_START[name], L.TOP + L.BLOCK_SIZE[2] / 2), size=L.BLOCK_SIZE, mass=L.BLOCK_MASS,
        color=L.BLOCK_BLUE)
    for name in L.BLOCKS
]
OBJECTS = [*_lens_parts("left"), *_lens_parts("right"), *BLOCK_OBJECTS]

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


def _blocks(objects) -> dict:
    try:
        return {name: objects.get(name) for name in L.BLOCKS}
    except AttributeError:
        return {}


def success(objects) -> bool:
    """Both lenses blocked on target: on each, exactly one block flat (within 3 deg), seated (lowest corner within
    2 mm of the lens top), still (path 2 mm, turn 2 deg in 0.5 s), let go (no gripper under half open with its TCP
    within 40 mm), its centre within 2 mm of the point 8 mm temporal of the OC and its long side within 3 deg of the
    line; held 1 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate read
    for this step)."""
    blocks = _blocks(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(blocks, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_opening=arms["left"].opening, right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    row["block_pose"] = {n: ((s.pos, s.quat_xyzw) if s is not None and hasattr(s, "pos") else None)
                         for n, s in blocks.items()}
    _trace(row)
    return result.success


def score(objects) -> float:
    """Left lens + right lens, each 0.5 (blocked, d <= 2 mm and a <= 3 deg), 0.25 (blocked, d <= 3 mm and
    a <= 6 deg) or 0: 0, 0.25, 0.5, 0.75 or 1.0."""
    return _TRACKER.result(_blocks(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part. Both blocks
# are gripped across their 18 mm side with the fingers closing along x (yaw -pi/2 left, +pi/2 right, the signs that
# keep joint 6 inside its limits), so the 24 mm side lies along y between the pads, as it stands.
OPEN = -0.04695
GRIP = -(L.BLOCK_SIZE[0] - 0.008) / 2  # -0.005: a 10 mm gap on the 18 mm side
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
PICK_H = 0.018  # TCP 2 mm above the block's top: the pad tips 5 mm above the table, 11 mm of the block held
ENTRY_H = 0.045  # the last 27 mm of the descent onto the block are slow
CARRY_H = 0.080  # the carried block's bottom 62 mm above the table
PLACE_ENTRY_H = 0.060
# Set down: the block's bottom (18 mm below the TCP) 1.0 mm above the marks' tops (12.1 mm), the pad tips 6 mm
# above the lens.
PLACE_H = L.MARK_TOP_H + 0.001 + PICK_H  # 0.0311
UP_H = 0.100
PICK_YAW = {"left": -math.pi / 2, "right": math.pi / 2}
# Per-arm corrections from the first GPU run (job 20261106T062247-00, object_jitter 0, the layout fixed), read from
# TASK_TRACE at the end, the blocks let go and still. Block centre minus target: left (-0.60, +1.24) mm (d 1.38),
# right (-0.94, -2.15) mm (d 2.34: the 0.25 tier, total 0.75). About 0.9 mm of each is the grasp (the TCP stopped
# 0.9 mm inboard of the block at the pick, and the fingers centre it only along their closing axis, x); the rest is
# the arm trailing its command at the place (left TCP (-0.2, +0.4) mm, right (-0.5, -1.3) mm). The offline check has
# neither, so it sees each trim as the block's whole error: the left trim is the full offset (offline d 1.34 mm),
# the right one 1.57 mm of its 2.34 (offline d 1.57 mm, about 0.8 mm expected on the GPU). Long side to the line:
# held just before the release, both blocks sat past the line (left +1.46 deg, right +1.84 deg: the grip turns the
# block anticlockwise); opening the fingers turned them back clockwise by 2.24 deg (left) and 0.23 deg (right), to
# a final -0.78 and +1.61 deg. That release turn is not one fixed number, so each yaw trim centres the outcomes over
# the 0.23 to 2.24 deg seen (-0.2 left, -0.6 right): a about 1.0 deg either way, 1.26 at most with no release turn,
# where trims that cancel the one sample per arm would reach 2.26 deg if the turns swapped or stopped.
PLACE_TRIM = {"left": (0.0006, -0.0012), "right": (0.0007, 0.0014)}  # m, added to the place point
PLACE_YAW_TRIM_DEG = {"left": -0.2, "right": -0.6}  # added to the place yaw
# Turning the gripper turns the block with it: from the 24 mm side along y (heading 90 deg) to the line's heading
# (115 deg left, 60 deg right): left -90 -> -65 deg, right +90 -> +60 deg, before the yaw trims (-65.2, +59.4).
PLACE_YAW = {s: PICK_YAW[s] + math.radians(L.line_heading_deg(s) - 90.0 + PLACE_YAW_TRIM_DEG[s]) for s in L.SIDES}
START = {"left": "block_left", "right": "block_right"}


def _arm(side: str) -> list[Waypoint]:
    """Pick this side's block, carry it over this side's lens turning it onto the line, set it down, let go."""
    bx, by = L.BLOCK_START[START[side]]
    tx, ty = L.TARGET[side]
    tx, ty = tx + PLACE_TRIM[side][0], ty + PLACE_TRIM[side][1]
    yp, yq = PICK_YAW[side], PLACE_YAW[side]
    z = L.TOP
    return [
        Waypoint(3.0, (bx, by, z + UP_H), OPEN, "over the block", yp),
        Waypoint(1.0, (bx, by, z + ENTRY_H), OPEN, "down toward it", yp),
        Waypoint(1.5, (bx, by, z + PICK_H), OPEN, "down beside it, slowly", yp),
        Waypoint(0.5, (bx, by, z + PICK_H), OPEN, "settle", yp),
        Waypoint(1.5, (bx, by, z + PICK_H), GRIP, "grip it across its 18 mm side", yp),
        Waypoint(1.5, (bx, by, z + CARRY_H), GRIP, "lift it", yp),
        Waypoint(2.5, (tx, ty, z + CARRY_H), GRIP, "carry it over the target, turning it onto the line", yq),
        Waypoint(1.0, (tx, ty, z + PLACE_ENTRY_H), GRIP, "down to 3 cm above the lens", yq),
        Waypoint(2.0, (tx, ty, z + PLACE_H), GRIP, "lower it the last 29 mm slowly", yq),
        Waypoint(0.7, (tx, ty, z + PLACE_H), GRIP, "settle", yq),
        Waypoint(1.5, (tx, ty, z + PLACE_H), OPEN, "let go: it drops 1 mm onto the lens", yq),
        Waypoint(1.5, (tx, ty, z + UP_H), OPEN, "straight up", yq),
    ]


def demo():
    """Both arms on one clock, mirror plans: each blocks the lens in front of it. DEMO_HOLD_ONLY=1: no plan (both
    arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm
    check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
