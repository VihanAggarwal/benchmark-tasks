"""On the station table: a loose pale grey litter (a flat board 500 mm long along y, 40 mm wide, 20 mm thick, 0.20 kg)
lying across the midline 24 cm in front of the arm bases with a loose orange 'casualty' (a 300 x 70 x 40 mm box,
0.30 kg) lying on its middle, and a fixed dark grey trolley deck (a 120 x 360 x 30 mm platform) 14 cm further
forward. Two-bearer rule: the litter is lifted by both handles at once, kept level, carried, and lowered onto the
trolley with both ends together; it is never dragged, never lifted at one end, and the casualty stays on it.

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
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long with both objects still. Longer than the demo's release (1.5 s) and rise (1.0 s), so
# the trial cannot end while a gripper still holds the litter on the trolley: the arm's sag (about 7 mm under its
# 0.25 kg share) sets the litter on the trolley before the slow descent ends, and the 1.0 conditions need both
# grippers let go, so the 3 s start counting only once the fingers have opened.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Lift the grey litter by both handles at the same time, keep it level, carry it forward and lower it onto "
    "the dark grey trolley, both ends together, then let go. Do not drag the litter and do not lift one end "
    "before the other. The orange casualty must stay on the litter."
)

LITTER = Box(
    "litter",
    pos=(L.LITTER_X, L.LITTER_Y, L.TOP + L.LITTER_SIZE[2] / 2),
    size=L.LITTER_SIZE,
    mass=L.LITTER_MASS,
    color=L.PALE_GREY,
)
CASUALTY = Box(
    "casualty",
    pos=(L.CASUALTY_X, L.CASUALTY_Y, L.CASUALTY_BOTTOM + L.CASUALTY_SIZE[2] / 2),
    size=L.CASUALTY_SIZE,
    mass=L.CASUALTY_MASS,
    color=L.ORANGE,
)
TROLLEY = Box(
    "trolley",
    pos=(L.TROLLEY_X, L.TROLLEY_Y, L.TOP + L.TROLLEY_SIZE[2] / 2),
    size=L.TROLLEY_SIZE,
    color=L.DARK_GREY,
)
OBJECTS = [LITTER, CASUALTY, TROLLEY]

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
    """Litter flat and centred on the trolley with the casualty on it, both grippers let go, the ends never more
    than 20 mm apart in height and the litter never tilted more than 5 degrees while handled, never dragged, never
    lifted at one end; held still 3 s. This runs first every control step, so it also feeds the tracker (with the
    arms rigstate read for this step)."""
    litter, casualty = objects["litter"], objects["casualty"]
    arms = rigstate.arms()
    result = _TRACKER.update(litter, casualty, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "litter_pos": litter.pos, "casualty_pos": casualty.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if at any moment the litter was dragged (more than 15 mm while within 10 mm of its surface), the casualty
    left the litter or tilted more than 15 degrees, the litter tilted more than 10 degrees, its ends differed by
    more than 40 mm in height while an end was up, or it was in the air without both grippers holding it; or if
    either object ends off the table. Else 1.0 litter flat on the trolley, centred within 15 mm (x) and 20 mm (y),
    casualty on it, both grippers let go, ends within 20 mm and tilt within 5 degrees throughout, lowered at
    0.10 m/s or less; 0.75 flat on the trolley within 40 mm with the casualty on it but not all of that; 0.5 on the
    trolley (tilted under 10 degrees) with the casualty on it; 0.25 carried level at some moment but not on the
    trolley; 0 otherwise."""
    return _TRACKER.result(objects["litter"], objects["casualty"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.LITTER_SIZE[0] - 0.008) / 2  # 32 mm commanded across the litter's 40 mm width
# Both grips close the fingers along x (yaw -pi/2 left, +pi/2 right, the signs that reach the midline), so the open
# fingers (94 mm apart) straddle the litter along x, clear of the casualty (46 mm away along y) and of the trolley
# (its near edge 25 mm beyond the open pad at the pick, and beside the pads, not under them, at the set-down).
YAW_L = -math.pi / 2
YAW_R = math.pi / 2
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.004  # the litter is gripped with the finger tips 4 mm above its bottom: 16 of its 20 mm held
GRIP_TCP_Z = L.TOP + GRIP_UP + PAD_TIPS  # TOP + 0.017
APPROACH = 0.030  # the final descent to the grip starts this far above it and takes 1.5 s
HOVER = L.TOP + 0.12  # over the handles before the pick and after the release
# Carry height: the litter's underside 63 mm above the table as commanded, which clears the 30 mm trolley by 33 mm.
# The arms sag under the load (joints 1-3: 80 N m/rad, no gravity compensation for the payload; each arm carries
# 0.25 kg, half the litter and casualty): about 7 mm down at this reach, and 1-2 deg of gripper tilt, so the
# underside is about 56 mm up in the simulator, 26 mm over the trolley; 'carried level' needs 30 mm over the table.
CARRY_Z = L.TOP + 0.080
CARRY_X = L.TROLLEY_X  # the carry ends with the litter centred over the trolley: 14 cm forward
# Set-down: let go with the litter's underside 2 mm above the trolley's top as commanded. With the sag the litter
# already rests on the trolley (the arm still carrying part of its weight) when the fingers open: no drop.
SET_TCP_Z = L.TROLLEY_TOP + 0.002 + GRIP_UP + PAD_TIPS
ENTRY_Z = SET_TCP_Z + 0.020  # the last 20 mm are lowered slowly
UP_Z = L.TOP + 0.10  # where each hand goes after letting go


def _bearer(y: float, yaw: float, side: str) -> list[Waypoint]:
    """One bearer's plan: grip the handle at (LITTER_X, y) across x, lift to the carry height, carry 14 cm forward,
    lower onto the trolley, let go, go up. Both bearers run the same timings on one clock."""
    x0, x1 = L.LITTER_X, CARRY_X
    return [
        Waypoint(1.5, (x0, y, HOVER), OPEN, f"over the {side} handle", yaw),
        Waypoint(1.0, (x0, y, GRIP_TCP_Z + APPROACH), OPEN, "down to 3 cm above the grip", yaw),
        Waypoint(1.5, (x0, y, GRIP_TCP_Z), OPEN, "slowly down beside the handle", yaw),
        Waypoint(0.5, (x0, y, GRIP_TCP_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (x0, y, GRIP_TCP_Z), GRIP, "grip the handle", yaw),
        Waypoint(0.5, (x0, y, GRIP_TCP_Z), GRIP, "settle the grip", yaw),
        Waypoint(2.0, (x0, y, CARRY_Z), GRIP, "lift together, straight up", yaw),
        Waypoint(3.0, (x1, y, CARRY_Z), GRIP, "carry forward over the trolley, level", yaw),
        Waypoint(1.0, (x1, y, ENTRY_Z), GRIP, "down to 2 cm above the set-down", yaw),
        Waypoint(0.5, (x1, y, ENTRY_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (x1, y, SET_TCP_Z), GRIP, "lower the last 2 cm together", yaw),
        Waypoint(0.5, (x1, y, SET_TCP_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (x1, y, SET_TCP_Z), OPEN, "let go", yaw),
        Waypoint(1.0, (x1, y, UP_Z), OPEN, "up", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: the left handle (y +0.21)."""
    return _bearer(L.HANDLE_Y, YAW_L, "left")


def _right() -> list[Waypoint]:
    """Right arm: the right handle (y -0.21), the mirror image on the same clock."""
    return _bearer(-L.HANDLE_Y, YAW_R, "right")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
