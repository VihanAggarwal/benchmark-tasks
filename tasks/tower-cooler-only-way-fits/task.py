"""On the station table: a bench model of a motherboard at about half scale (a green board 160 x 170 mm, fixed),
centred 30 cm in front of the arm bases with its rear I/O edge away from the arms. On it, all fixed: the processor
(a grey 38 x 38 mm lid) in a black retention frame; two tall black memory modules 27 mm from the processor's
centre on its near side; a dark grey VRM heatsink on its left; a black graphics card standing in the slot on its
right; the grey rear I/O shield along the far edge. On a pink foam pad to the left lies the loose tower cooler: a
copper base, a copper riser and a 74 mm wide aluminium fin stack with a black fan on one face, the stack offset
along the airflow so it reaches 37 mm from the base's centre one way and 18 mm the other. It stands with its airflow
along +y.
Rule: the cooler's base must end flat on the processor inside the frame, which is possible only with its airflow
toward the rear exhaust (+x); turned any other way, the fin stack lands on the memory.

The cooler is a loose mechkit compound (mechkit.py, next to this file): one rigid body of four boxes, so the scorer
can read its pose (base frame and orientation). The scorer also reads the arms (TCP, finger tips, gripper opening)
through rigstate.py, which wraps the embodiment's step and reset from this file's import; neither changes anything
the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the task needs only one; the demo uses the left arm and the right arm holds its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the cooler is built as a compound body)
import rigstate  # noqa: E402
from scoring import Tracker, to_world  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with both grippers let go (open, opening >= 0.9, or the finger zone > 10 mm from the cooler): the
# trial cannot end while the gripper still holds the cooler.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Mount the tower cooler on the processor: set it down so its copper base sits flat on the processor inside the "
    "black retention frame, then let go. On this board it fits only one way round."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


def _centred_box(name, x, y, size, z0, color) -> Box:
    """A fixed box centred at (x, y) seen from above, standing on height z0 (above the table top)."""
    return Box(name, pos=(x, y, L.TOP + z0 + size[2] / 2), size=size, color=color)


BOARD = _span_box("motherboard", L.BOARD_X0, L.BOARD_X1, L.BOARD_Y0, L.BOARD_Y1, 0.0, L.BOARD_T, L.PCB_GREEN)
CPU = _centred_box("processor", L.CPU_X, L.CPU_Y, L.CPU_SIZE, L.BOARD_TOP, L.LID_GREY)
_o, _w = L.FRAME_OPEN_HALF, L.FRAME_BAR_W
FRAME = [  # four bars round the 44 x 44 mm opening; the near and far bars run the full width
    _span_box("frame_near", L.CPU_X - _o - _w, L.CPU_X - _o, L.CPU_Y - _o - _w, L.CPU_Y + _o + _w,
              L.BOARD_TOP, L.FRAME_TOP, L.FRAME_BLACK),
    _span_box("frame_far", L.CPU_X + _o, L.CPU_X + _o + _w, L.CPU_Y - _o - _w, L.CPU_Y + _o + _w,
              L.BOARD_TOP, L.FRAME_TOP, L.FRAME_BLACK),
    _span_box("frame_left", L.CPU_X - _o, L.CPU_X + _o, L.CPU_Y + _o, L.CPU_Y + _o + _w,
              L.BOARD_TOP, L.FRAME_TOP, L.FRAME_BLACK),
    _span_box("frame_right", L.CPU_X - _o, L.CPU_X + _o, L.CPU_Y - _o - _w, L.CPU_Y - _o,
              L.BOARD_TOP, L.FRAME_TOP, L.FRAME_BLACK),
]
MEMORY = [_centred_box(f"memory_{i + 1}", x, L.CPU_Y, L.MEM_SIZE, L.BOARD_TOP, L.MEM_BLACK)
          for i, x in enumerate(L.MEM_X)]
VRM = _centred_box("vrm_heatsink", L.VRM_X, L.VRM_Y, L.VRM_SIZE, L.BOARD_TOP, L.VRM_GREY)
GPU = _centred_box("graphics_card", L.GPU_X, L.GPU_Y, L.GPU_SIZE, L.BOARD_TOP, L.GPU_BLACK)
IO = _centred_box("io_shield", L.IO_X, L.IO_Y, L.IO_SIZE, L.BOARD_TOP, L.IO_GREY)
PAD = _centred_box("foam_pad", L.PAD_X, L.PAD_Y, L.PAD_SIZE, 0.0, L.FOAM_PINK)
COOLER = mk.assembly(
    "cooler",
    origin=L.START,
    yaw=L.START_YAW,
    mass=L.COOLER_MASS,
    color=L.COPPER,
    shapes=[
        mk.box(L.BASE_CENTER, L.BASE_SIZE),
        mk.box(L.RISER_CENTER, L.RISER_SIZE),
        mk.box(L.FAN_CENTER, L.FAN_SIZE, color=L.FAN_BLACK),
        mk.box(L.FINS_CENTER, L.FINS_SIZE, color=L.ALUMINIUM),
    ],
)
OBJECTS = [BOARD, CPU, *FRAME, *MEMORY, VRM, GPU, IO, PAD, COOLER]

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
    """The cooler's base flat on the processor inside the frame (within 2 mm of the lid, tilted 2 degrees or less,
    its four corners inside the 44 x 44 mm opening), its airflow within 15 degrees of the rear exhaust, at rest
    (base centre and fin-stack top each moved 2 mm or less in 0.5 s), both grippers let go; held 1 s. This runs
    first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    cooler = objects["cooler"]
    arms = rigstate.arms()
    result = _TRACKER.update(cooler, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(cooler, "source", None), cooler_origin=cooler.origin, cooler_quat=cooler.quat_xyzw,
                   cooler_vel=cooler.lin_vel, stack_top=to_world(cooler, L.TOP_LOCAL))
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the cooler is at rest. 1.0 seated (base flat on the processor inside the frame, airflow toward the
    rear exhaust) and let go; 0.5 seated but still held, or let go, upright within 30 degrees, the right way round
    (airflow within 15 degrees of +x) and its base centre within 20 mm of the processor's centre but not seated;
    0.25 let go, upright and its base centre over the board, but turned another way round (e.g. on the memory) or
    further off; 0 otherwise (still held, tipped over, not on the board). Read from the state at hand."""
    return _TRACKER.result(objects["cooler"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.UPSTREAM + L.DOWNSTREAM - 0.008) / 2  # across fan plus fins, 55 mm along the airflow: 47 mm commanded
# The fingers close along the airflow, the pads on the fan's intake face and the fins' outlet face, centred on the
# middle of fan plus fins (9.5 mm downstream of the base's centre). Across the airflow the cooler is symmetric, so
# it hangs square in the grip; its centre of mass (42.8 mm up) is level with the pads (37-70 mm up).
# Picked with the fingers along y (yaw 0, the airflow is +y), turned a quarter turn as it is carried, set down with
# the fingers along x: the left arm's yaw -pi/2 (the other sign runs joint6 into its limit near the midline). The
# turn takes the airflow from +y to +x.
YAW_PICK, YAW_PLACE = 0.0, -math.pi / 2
GRIP_UP = 0.050  # the TCP 50 mm above the base's bottom face: pad tips 37 mm up, 17 mm above the fins' foot
DROP = 0.001  # let go with the base 1 mm above the lid (the arm sags a few mm under the 0.25 kg cooler)
APPROACH = 0.030  # the last 30 mm of each descent are slow
CARRY_Z = L.TOP + 0.11  # carried with the base 60 mm up: 26 mm over the memory, 30 mm over the VRM heatsink
PICK = (L.PAD_X, L.PAD_Y + L.STACK_MID, L.TOP + L.PAD_TOP + GRIP_UP)
# Trim of the place point (x, y) for the arm's sideways sag under the load. In an earlier GPU run (L40S), a 0.20 kg
# block put down by the left arm at (0.30, 0) landed (-1.8, +1.6) mm off its aim; this trims by about that, scaled to
# 0.25 kg and rounded down. Offline (no sag) the base lands 2.1 mm off centre,
# well inside the frame's 5 mm. If the GPU trace shows base_dx_mm / base_dy_mm away from 0, move the trim by minus
# the offset.
PLACE_TRIM = (0.0015, -0.0015)
PLACE = (L.CPU_X + L.STACK_MID + PLACE_TRIM[0], L.CPU_Y + PLACE_TRIM[1], L.TOP + L.CPU_TOP + DROP + GRIP_UP)
PARK = (0.22, 0.20, L.TOP + 0.15)


def _left() -> list[Waypoint]:
    """Left arm: grip the cooler across fan and fins, lift it, carry it over the processor while turning it a quarter
    turn (airflow from +y to +x), lower it slowly onto the lid, let go, go up and aside."""
    px, py, pz = PICK
    qx, qy, qz = PLACE
    return [
        Waypoint(2.0, (px, py, CARRY_Z), OPEN, "over the cooler", YAW_PICK),
        Waypoint(1.0, (px, py, pz + APPROACH), OPEN, "down to just above the grip", YAW_PICK),
        Waypoint(1.5, (px, py, pz), OPEN, "slowly down beside fan and fins", YAW_PICK),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (px, py, pz), GRIP, "grip the cooler", YAW_PICK),
        Waypoint(1.5, (px, py, CARRY_Z), GRIP, "lift it off the pad", YAW_PICK),
        Waypoint(2.5, (qx, qy, CARRY_Z), GRIP, "carry it over the processor, turning the airflow to the rear",
                 YAW_PLACE),
        Waypoint(0.8, (qx, qy, CARRY_Z), GRIP, "settle over the processor", YAW_PLACE),
        Waypoint(1.0, (qx, qy, qz + APPROACH), GRIP, "down to 3 cm above", YAW_PLACE),
        Waypoint(2.5, (qx, qy, qz), GRIP, "lower the base slowly onto the lid", YAW_PLACE),
        Waypoint(0.5, (qx, qy, qz), GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (qx, qy, qz), OPEN, "let go of the cooler", YAW_PLACE),
        Waypoint(1.0, (qx, qy, CARRY_Z), OPEN, "straight up", YAW_PLACE),
        Waypoint(1.5, PARK, OPEN, "aside", YAW_PLACE),
    ]


def demo():
    """One arm: the right arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
