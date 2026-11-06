"""On the station table: a grey jig plate (300 x 200 x 6 mm, fixed) centred on the midline 15-45 cm in front of the
arm bases, with an inlaid red laser line across it at x 0.270 and a dark centre line along it (y 0); an orange heel
block (60 x 70 x 20 mm) on the table in front of the left arm; and a right prosthetic foot (a 220 mm flat-soled
shell with a black adapter block on top, 0.25 kg) lying sole down in front of the right arm, toes forward. Bench
alignment: the heel block goes flat on the plate under the heel; the foot goes on, heel on the block and toe on
the plate, the middle of its length 30 mm in front of the laser line, the heel on the centre line, toes turned out
5 degrees toward -y (the right foot's outside), and is let go, both grippers moved clear.

The foot is a mechkit assembly (mechkit.py, next to this file: one loose rigid body of three boxes); the scorer
reads its pose from it, and the arms (tool point, pad-tip point) through rigstate.py, which wraps the embodiment's
step and reset from this file's import. Neither changes what the embodiment does.

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

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the foot is built as one compound body)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from standin import rest_pose  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success needs the gate (both arms' tool and pad-tip points 20 mm clear of the foot and the block, everything still
# over 0.5 s) and all three checks, held for 1 s: the trial cannot end while a gripper still holds anything.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Bench-align the right prosthetic foot on the grey jig plate. First put the orange heel block flat on the plate "
    "where the heel will rest. Then set the foot on the plate, sole down, with its heel on the block and its toe on "
    "the plate: the middle of the foot's length 30 mm in front of the red laser line, the heel on the dark centre "
    "line, and the toe end turned out about 5 degrees toward your right. Then let go and move both grippers clear "
    "of the foot and the block."
)


class Foot(mk.Assembly):
    """The foot exactly as mechkit builds and reads it in the simulator. Only the offline checker calls the two
    methods below: it draws and grips the foot as the adapter's column (46 x 46 mm, sole to the adapter's top), the
    part a gripper holds, instead of the 220 x 70 mm box round the whole foot, and the hooks get the foot's frame
    rebuilt from that column (see standin.py for how it then comes to rest)."""

    def offline_box(self):
        c, s = L.COLUMN_CENTER, L.COLUMN_SIZE
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        ox, oy, oz = self.pos
        return (ox + cy * c[0] - sy * c[1], oy + sy * c[0] + cy * c[1], oz + c[2]), tuple(s), float(self.yaw)

    def offline_state(self, state):
        r = mk.quat_to_matrix(state.quat_xyzw)
        origin = np.asarray(state.pos, dtype=float) - r @ np.asarray(L.COLUMN_CENTER, dtype=float)
        vel = (*state.lin_vel, *state.ang_vel)
        return mk.state_from_poses(self, (*origin, *state.quat_xyzw), vel, source="offline")


def _plate_piece(name, xs, ys, colour) -> Box:
    (x0, x1), (y0, y1) = xs, ys
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.PLATE_T / 2), size=(x1 - x0, y1 - y0, L.PLATE_T),
               color=L.PIECE_COLOURS[colour])


PLATE = [_plate_piece(*p) for p in L.PLATE_PIECES]
BLOCK = Box("heel_block", pos=(*L.BLOCK_START, L.TOP + L.BLOCK_HALF[2]), size=L.BLOCK_SIZE, mass=L.BLOCK_MASS,
            color=L.BLOCK_ORANGE)
FOOT = Foot(name="foot", pos=L.FOOT_ORIGIN, color=L.FOOT_BEIGE, mass=L.FOOT_MASS,
            shapes=(mk.box(L.REAR_CENTER, L.REAR_SIZE), mk.box(L.FORE_CENTER, L.FORE_SIZE),
                    mk.box(L.ADAPTER_CENTER, L.ADAPTER_SIZE, color=L.ADAPTER_BLACK)))
FOOT._check()
OBJECTS = [*PLATE, BLOCK, FOOT]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


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


def _observe(objects):
    """(foot, block) for the scorer. Offline (mechkit marks the foot's state 'offline': the checker carries it level
    and drops it flat) the stand-in's resting pose replaces it; in the simulator nothing is changed."""
    foot, block = objects["foot"], objects["heel_block"]
    if getattr(foot, "source", None) == "offline":
        try:
            origin, quat = rest_pose(foot.origin, foot.quat_xyzw, block)
            foot = mk.state_from_poses(FOOT, (*origin, *quat), None, source="offline")
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            pass  # a malformed offline state: the scorer's own checks give 0
    return foot, block


def success(objects) -> bool:
    """The gate (sole over the plate, sole down within 15 deg, toes forward within 20 deg, heel point within 15 mm
    of the centre line, both grippers 20 mm clear, all still over 0.5 s) and all three checks (the foot's middle
    25-35 mm in front of the laser line, toe-out 2-8 deg, heel on the block lying flat on the plate, no corner of
    the block more than 90 mm in front of the heel point, and the toe on the plate), held 1 s. This runs first
    every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    foot, block = _observe(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(foot, block, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(foot, "source", None), foot_origin=foot.origin, foot_quat=foot.quat_xyzw,
                   block_pos=block.pos, block_quat=block.quat_xyzw,
                   block_shift_mm=round(1000 * math.hypot(float(block.pos[0]) - BLOCK_PLACE[0],
                                                          float(block.pos[1]) - BLOCK_PLACE[1]), 1))
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips, opening={a: round(float(arms[a].opening), 3) for a in arms})
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the gate holds: all four sole corners over the plate, the foot sole down (up axis within 15 deg of
    vertical), heel to toe within 20 deg of +x, the heel point within 15 mm of the centre line, both arms' tool and
    pad-tip points 20 mm or more from the foot and the block, and the heel point, toe point and block each moved
    less than 2 mm over the last 0.5 s. Then 0.25, plus 0.25 each for: the foot's middle 25-35 mm in front of the
    laser line; toe-out 2-8 deg toward -y; the heel on the block (block flat on the plate, under the heel with no
    corner more than 90 mm in front of the heel point, heel point 18-40 mm up, toe point within 4 mm of the plate)."""
    foot, block = _observe(objects)
    return _TRACKER.result(foot, block, rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
BLOCK_GRIP = -(L.BLOCK_SIZE[1] - 0.008) / 2  # yaw 0: across the block's 70 mm (y)
ADAPTER_GRIP = -(L.ADAPTER_SIZE[1] - 0.008) / 2  # yaw 0: across the adapter's 46 mm sides (the foot's width)
TOE_OUT = math.radians(5.0)
YAW_PLACE = -TOE_OUT  # the right gripper turns -5 deg on the way, and the foot with it
GRIP_UP = 0.008  # the pad tips this far above what they close on: the block's bottom, the rear shell's top


def _z(h: float) -> float:
    return L.TOP + h


# Left arm and the block: tips 8 mm above its bottom (12 of its 20 mm held); set down 1 mm above the plate.
BLOCK_PICK_Z = _z(GRIP_UP + L.PAD_TIPS)  # 0.021 above the table
BLOCK_PLACE = (0.220, 0.000)  # the block spans x 0.190-0.250 under the heel; its centre 26 mm along the foot from H
BLOCK_PLACE_Z = L.PLATE_TOP + 0.001 + GRIP_UP + L.PAD_TIPS  # 0.028 above the table
BLOCK_CARRY_Z = _z(0.080)  # the block's bottom 59 mm up
PARK_L = (0.250, 0.300, _z(0.150))

# Right arm and the foot: gripped on the adapter's two sides (fingers closing along y, across the foot) with the TCP
# 49 mm above the sole, the pad tips 36 mm up: 8 mm above the rear shell's 28 mm top, 22 mm of the 30 mm adapter
# held. Fingers along y, the foot's pivot onto the block at release runs along the pad faces, not into a pad.
GRIP_TCP_UP = L.REAR_SIZE[2] + GRIP_UP + L.PAD_TIPS  # 0.049 above the sole
FOOT_PICK = (L.FOOT_ORIGIN[0] + L.ADAPTER_CENTER[0], L.FOOT_ORIGIN[1])  # (0.275, -0.380)
FOOT_CARRY_Z = _z(0.120)  # the sole 71 mm up: over the block's 26 mm top with 45 mm to spare
# The target: H on the centre line, M = H + 110 mm along the foot 30 mm in front of the laser line, toes out 5 deg.
HEEL_TARGET = (L.LASER_X + L.AP_TARGET - (L.FOOT_LEN / 2) * math.cos(TOE_OUT), 0.0)  # (0.1904, 0)
# Payload sag: the arm's controller does not carry the foot's weight. The static estimate (J K^-1 J^T of the 0.25 kg
# weight and its 9 mm lever; joints 80/80/80/10/10/10 N m/rad) at the place is (-2.2, -1.9, -6.6) mm and 0.9 deg
# toe down. Along x it would set the foot's middle 2.2 mm short, so PLACE_TRIM aims 3.0 mm forward: offline (no sag)
# the middle lands at 32.6 mm (the pivot onto the block's edge takes 0.4 mm back), with the estimated sag at 30.4 mm,
# and with anything from no sag to twice the estimate at 28.2 to 32.6 mm, at least 2.4 mm inside the 25-35 mm band.
# Sideways the trim takes back about half of the -1.9 mm (the heel's band is +-15 mm). On the GPU, read ap_mm and
# heel_y_mm in TASK_TRACE and adjust.
PLACE_TRIM = (0.0030, 0.0010)
FOOT_PLACE = (HEEL_TARGET[0] + L.ADAPTER_CENTER[0] * math.cos(YAW_PLACE) + PLACE_TRIM[0],
              HEEL_TARGET[1] + L.ADAPTER_CENTER[0] * math.sin(YAW_PLACE) + PLACE_TRIM[1])  # (0.2781, -0.0064)
# Let go level with the sole 7 mm above the block's top, more than the estimated 6.6 mm vertical sag (the 0.9 deg
# toe-down sag lifts the sole over the block's front edge 0.4 mm), so by the estimate the sole is still 0.4 to 0.8 mm
# clear of the block when the fingers open: the heel drops onto the block, the foot tips toe-down about the block's
# front edge and its toe end drops about 20 mm more to the plate (7.2 deg). Watch block_shift_mm in TASK_TRACE.
RELEASE_GAP = 0.007
FOOT_PLACE_Z = L.PLATE_TOP + L.BLOCK_SIZE[2] + RELEASE_GAP + GRIP_TCP_UP  # 0.082 above the table
PARK_R = (0.250, -0.300, _z(0.150))


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left() -> list[Waypoint]:
    """Left arm: the heel block from the table to the plate, under where the heel will be; then out of the way."""
    bx, by = L.BLOCK_START
    px, py = BLOCK_PLACE
    return [
        Waypoint(3.0, (bx, by, _z(0.100)), OPEN, "over the heel block"),
        Waypoint(1.0, (bx, by, BLOCK_PICK_Z + 0.030), OPEN, "down beside it"),
        Waypoint(1.5, (bx, by, BLOCK_PICK_Z), OPEN, "down the last 3 cm slowly"),
        Waypoint(0.5, (bx, by, BLOCK_PICK_Z), OPEN, "settle"),
        Waypoint(1.5, (bx, by, BLOCK_PICK_Z), BLOCK_GRIP, "grip the block across its 70 mm"),
        Waypoint(1.5, (bx, by, BLOCK_CARRY_Z), BLOCK_GRIP, "lift"),
        Waypoint(2.5, (px, py, BLOCK_CARRY_Z), BLOCK_GRIP, "carry it over the plate"),
        Waypoint(1.0, (px, py, BLOCK_PLACE_Z + 0.030), BLOCK_GRIP, "down to 3 cm above"),
        Waypoint(1.5, (px, py, BLOCK_PLACE_Z), BLOCK_GRIP, "lower the last 3 cm slowly"),
        Waypoint(0.5, (px, py, BLOCK_PLACE_Z), BLOCK_GRIP, "settle"),
        Waypoint(1.5, (px, py, BLOCK_PLACE_Z), OPEN, "let go of the block"),
        Waypoint(1.5, (px, py, _z(0.100)), OPEN, "straight up"),
        Waypoint(3.0, PARK_L, OPEN, "out of the way"),
    ]


def _right_pick() -> list[Waypoint]:
    fx, fy = FOOT_PICK
    pick_z = _z(GRIP_TCP_UP)
    return [
        Waypoint(3.0, (fx, fy, FOOT_CARRY_Z), OPEN, "over the foot's adapter"),
        Waypoint(1.0, (fx, fy, pick_z + 0.030), OPEN, "down beside the adapter"),
        Waypoint(1.5, (fx, fy, pick_z), OPEN, "down the last 3 cm slowly"),
        Waypoint(0.5, (fx, fy, pick_z), OPEN, "settle"),
        Waypoint(1.5, (fx, fy, pick_z), ADAPTER_GRIP, "grip the adapter's sides"),
        Waypoint(1.5, (fx, fy, FOOT_CARRY_Z), ADAPTER_GRIP, "lift the foot level"),
    ]


def _right() -> list[Waypoint]:
    """Right arm: pick the foot up by its adapter, hold it up until the left arm has set the block down and left,
    then carry it over the plate turning it 5 deg, lower it level to 7 mm above the block, let go, go up."""
    fx, fy = FOOT_PICK
    px, py = FOOT_PLACE
    pick = _right_pick()
    wait = _span(_left()) - _span(pick)  # the left arm is out of the way (parked) at 20.5 s
    return pick + [
        Waypoint(wait, (fx, fy, FOOT_CARRY_Z), ADAPTER_GRIP, "hold it up until the block is down"),
        Waypoint(3.0, (px, py, FOOT_CARRY_Z), ADAPTER_GRIP, "carry it over the plate, toes turned out", YAW_PLACE),
        Waypoint(1.0, (px, py, FOOT_PLACE_Z + 0.020), ADAPTER_GRIP, "down to 2 cm above", YAW_PLACE),
        Waypoint(2.0, (px, py, FOOT_PLACE_Z), ADAPTER_GRIP, "lower the last 2 cm slowly", YAW_PLACE),
        Waypoint(0.5, (px, py, FOOT_PLACE_Z), ADAPTER_GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (px, py, FOOT_PLACE_Z), OPEN, "let go: heel onto the block, toe onto the plate", YAW_PLACE),
        Waypoint(1.5, (px, py, _z(0.130)), OPEN, "straight up", YAW_PLACE),
        Waypoint(3.0, PARK_R, OPEN, "out of the way"),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
