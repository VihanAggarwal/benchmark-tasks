"""On the station table: a wooden storage tray (white-painted plywood floor, dark-stained beech rims; 170 x 126 mm
inside, fixed) on the midline in front of the arm bases, its length pointing away from the arms; on each side of it a
steel bar magnet (100 x 20 x 15 mm, north half red, south half blue) lying across the table with its red end pointing
outward, the two a mirror image of each other; and nearer the arms, one on each side, a dark grey soft-iron keeper
(20 x 74 x 15 mm). Rule: the
magnets go into the tray side by side along its length with unlike poles beside each other (the red end of each next
to the blue end of the other), then a keeper across each end of the pair, touching both magnets. Like poles side by
side scores 0.

Each magnet is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of two boxes, so its two
halves can have their own colours and the scorer can read which way its red end points.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the magnets are built as compound bodies)
from scoring import KEEPERS, MAGNETS, Tracker, magnet_centre, red_sign  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The arm sags a few mm under the 0.17 kg keeper, so the last keeper comes to
# rest on the floor while the slow descent is still finishing and success starts counting (after the 0.5 s needed to
# see it at rest) with the gripper still closed. Settling, letting go and lifting take 0.5 + 1.5 + 1.0 s, so with
# 2.5 s the trial ends with the fingers open and the hand going up.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Put the two bar magnets away in the wooden tray as a pair: lying side by side along the tray, one end of each "
    "toward you and the other away from you, with the red (north) end of each next to the blue (south) end of the "
    "other. Then lay one grey iron keeper across each end of the pair, touching the ends of both magnets."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


_X0, _X1, _YI, _T = L.INNER_X0, L.INNER_X1, L.INNER_Y, L.RIM_T
TRAY = [
    _span_box("tray_floor", _X0 - _T, _X1 + _T, -_YI - _T, _YI + _T, 0.0, L.FLOOR_TOP, L.FLOOR_WHITE),
    _span_box("tray_rim_left", _X0 - _T, _X1 + _T, _YI, _YI + _T, L.FLOOR_TOP, L.RIM_TOP, L.RIM_STAIN),
    _span_box("tray_rim_right", _X0 - _T, _X1 + _T, -_YI - _T, -_YI, L.FLOOR_TOP, L.RIM_TOP, L.RIM_STAIN),
    _span_box("tray_rim_near", _X0 - _T, _X0, -_YI, _YI, L.FLOOR_TOP, L.RIM_TOP, L.RIM_STAIN),
    _span_box("tray_rim_far", _X1, _X1 + _T, -_YI, _YI, L.FLOOR_TOP, L.RIM_TOP, L.RIM_STAIN),
]


def _magnet(name: str, start, yaw: float):
    return mk.assembly(
        name,
        origin=(start[0], start[1], L.TOP + start[2]),
        yaw=yaw,
        mass=L.MAG_MASS,
        color=L.NORTH_RED,
        shapes=[
            mk.box(L.NORTH_CENTER, L.HALF_SIZE, color=L.NORTH_RED),
            mk.box(L.SOUTH_CENTER, L.HALF_SIZE, color=L.SOUTH_BLUE),
        ],
    )


MAGNET_A = _magnet("magnet_a", L.MAG_A_START, L.MAG_A_YAW)
MAGNET_B = _magnet("magnet_b", L.MAG_B_START, L.MAG_B_YAW)
KEEPER_1 = Box("keeper_1", pos=(*L.KEEP_1_START, L.TOP + L.KEEP_SIZE[2] / 2), size=L.KEEP_SIZE, mass=L.KEEP_MASS,
               color=L.IRON)
KEEPER_2 = Box("keeper_2", pos=(*L.KEEP_2_START, L.TOP + L.KEEP_SIZE[2] / 2), size=L.KEEP_SIZE, mass=L.KEEP_MASS,
               color=L.IRON)
OBJECTS = [*TRAY, MAGNET_A, MAGNET_B, KEEPER_1, KEEPER_2]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _states(objects) -> dict:
    return {n: objects[n] for n in (*MAGNETS, *KEEPERS)}


def success(objects) -> bool:
    """Both magnets stored side by side with unlike poles and a keeper closing each end, everything at rest; held
    2.5 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    states = _states(objects)
    result = _TRACKER.update(states)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        for n in MAGNETS:
            row[f"{n}_origin"] = states[n].origin
            row[f"{n}_quat"] = states[n].quat_xyzw
            row[f"{n}_source"] = getattr(states[n], "source", None)
            row[f"{n}_red_sign"] = red_sign(states[n])
            row[f"{n}_centre"] = magnet_centre(states[n])
        for n in KEEPERS:
            row[f"{n}_pos"] = states[n].pos
            row[f"{n}_quat"] = states[n].quat_xyzw
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if both magnets lie lengthwise in the tray with their red ends pointing the same way, or if no magnet is
    stored (flat on the floor, along the tray, inside the rims, at rest). 0.25 one stored, or both stored but not
    side by side (centres more than 60 mm apart across the tray or 15 mm along it). 0.5 a side-by-side pair with
    unlike poles; 0.75 and a keeper closing one end (on the floor, across, its near edge -2..8 mm from both end
    faces, spanning 12 mm of each); 1.0 both ends closed. Read from the state at hand (the last step)."""
    return _TRACKER.result(_states(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # 94 mm
MAG_OPEN = -0.025  # 50 mm: 15 mm either side of the 20 mm magnet on the way down
MAG_GRIP = -(L.MAG_W - 0.008) / 2  # 12 mm commanded across the magnet's 20 mm width
MAG_RELEASE = -0.018  # let go by opening to 36 mm only: 8 mm off each side, so the finger between the two magnets
#                       stays 9 mm clear of the one already in the tray
KEEP_GRIP = -(L.KEEP_SIZE[1] - 0.008) / 2  # 66 mm commanded across the keeper's 74 mm length
KEEP_RELEASE = -0.044  # 88 mm: the pads 7 mm off the keeper's ends, 11 mm inside the side rims
# The magnets are gripped at their middle across their 20 mm width (fingers closing along x) and turned a quarter
# turn clockwise (seen from above) in the hand while they are carried, so both arms turn the same way in the world:
# the left magnet's red end swings from +y to +x (away from the arms), the right one's from -y to -x. Turning the two
# the mirror-image way (left clockwise, right counter-clockwise) would put both red ends at the same end: 0.
YAW_PICK = math.pi / 2  # fingers closing along x, both arms
YAW_PLACE = 0.0  # fingers closing along y: the magnet now lies along x
GRIP_UP = 0.017  # magnets: the TCP 17 mm above the underside: pad tips 4 mm above it, 11 mm of the 15 mm held
# Keepers: the TCP 21 mm above the underside: pad tips 8 mm above it, 7 mm of the 15 mm held. A keeper overhangs the
# pair's outer faces by only 4.5 mm a side, so a few mm of sideways sag puts one of its pads in line with a magnet's
# end. The pads are about 27 mm wide at the TCP and 15 mm at their tips, so with the TCP this high it is their narrow
# lower part that comes level with the magnets' top corners (15 mm up) as a keeper is laid 4 mm off the end faces:
# a pad clears those corners unless the keeper lands within 0.7 mm of them (within 2.6 mm with the TCP at 17 mm).
KEEP_GRIP_UP = 0.021
DROP = 0.002  # parts are let go with their underside 2 mm above the tray floor (the arm sags a few mm under them)
APPROACH = 0.030  # each final descent starts this far above its end
CARRY = L.TOP + 0.12  # carried with a magnet's underside 103 mm up (a keeper's 99 mm), 87 mm or more over the rims
CARRY_FAR = L.TOP + 0.10  # lower near the far end of the tray, on the midline (the arm reaches x 0.35 there at 0.12)
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.FLOOR_TOP_W + DROP + GRIP_UP
KEEP_PICK_Z = L.TOP + KEEP_GRIP_UP
KEEP_PLACE_Z = L.FLOOR_TOP_W + DROP + KEEP_GRIP_UP
# Trims of the place points (x, y), for the GPU run: if a part lands off its aim, read the magnets' centres and the
# keepers' gaps (gaps_mm, aim 4 mm) in TASK_TRACE and move the place point by minus the offset.
TRIM_A = (0.0, 0.0)
TRIM_B = (0.0, 0.0)
TRIM_K_FAR = (-0.0035, 0.0)  # GPU run 20261107T043341: magnets landed ~4 mm short, keeper 1 far gaps 7.4/7.8 mm (band -2..8): place it 3.5 mm nearer
TRIM_K_NEAR = (0.0, 0.0)
PLACE_A = (L.PAIR_X + TRIM_A[0], L.PAIR_DY + TRIM_A[1])
PLACE_B = (L.PAIR_X + TRIM_B[0], -L.PAIR_DY + TRIM_B[1])
PLACE_K_FAR = (L.KEEP_FAR_X + TRIM_K_FAR[0], 0.0 + TRIM_K_FAR[1])
PLACE_K_NEAR = (L.KEEP_NEAR_X + TRIM_K_NEAR[0], 0.0 + TRIM_K_NEAR[1])
ASIDE_LEFT = (0.22, 0.30, L.TOP + 0.15)
ASIDE_RIGHT = (0.22, -0.30, L.TOP + 0.15)


def _pick(xy, open_f: float, grip_f: float, yaw: float, what: str, z: float = PICK_Z) -> list[Waypoint]:
    """From anywhere: over the part, down, slowly down beside it to the grip height ``z``, settle, grip, lift
    (7.2 s)."""
    x, y = xy
    return [
        Waypoint(1.5, (x, y, CARRY), open_f, f"over the {what}", yaw),
        Waypoint(1.0, (x, y, z + APPROACH), open_f, f"down to just above the {what}", yaw),
        Waypoint(1.5, (x, y, z), open_f, f"slowly down beside the {what}", yaw),
        Waypoint(0.5, (x, y, z), open_f, "settle", yaw),
        Waypoint(1.5, (x, y, z), grip_f, f"grip the {what}", yaw),
        Waypoint(1.2, (x, y, CARRY), grip_f, f"lift the {what}", yaw),
    ]


def _place(xy, carry_z: float, grip_f: float, release_f: float, yaw: float, what: str, carry_s: float = 2.5,
           z: float = PLACE_Z) -> list[Waypoint]:
    """From over its pick: carry over the place (turning to ``yaw``), settle, down, slowly down to the release height
    ``z``, settle, let go, straight up (carry_s + 6.8 s)."""
    x, y = xy
    return [
        Waypoint(carry_s, (x, y, carry_z), grip_f, f"carry the {what} over its place", yaw),
        Waypoint(0.8, (x, y, carry_z), grip_f, "settle over it", yaw),
        Waypoint(1.0, (x, y, z + APPROACH), grip_f, "down to 3 cm above the floor", yaw),
        Waypoint(2.0, (x, y, z), grip_f, f"slowly lay the {what} on the tray floor", yaw),
        Waypoint(0.5, (x, y, z), grip_f, "settle", yaw),
        Waypoint(1.5, (x, y, z), release_f, f"let go of the {what}", yaw),
        Waypoint(1.0, (x, y, carry_z), release_f, "straight up", yaw),
    ]


def _wait(until: float, plan: list[Waypoint], label: str) -> list[Waypoint]:
    """Hold the last pose until the plan's clock reaches ``until`` (s)."""
    t = sum(w.duration for w in plan)
    last = plan[-1]
    if until <= t + 1e-9:
        return []
    return [Waypoint(until - t, last.pos, last.finger, label, last.yaw)]


# The clock (s): left stores magnet A (0-16.5); right, holding magnet B since 7.2, stores it once the left hand is
# up (16.5-25.8); left fetches keeper 1 meanwhile and lays it across the far end once the right hand is up from B
# and back over keeper 2 (27.3-36.6: the forearms would pass within 2.5 cm if both moved at once); right fetches
# keeper 2 meanwhile and lays it across the near end once the left hand is up and going aside (36.6-45.9). The two
# hands are never over the tray at the same time.
T_A_UP = 16.5  # left hand up from magnet A
T_B_CLEAR = 27.3  # right hand up from magnet B and over keeper 2
T_K1_UP = 36.6  # left hand up from keeper 1


def _left() -> list[Waypoint]:
    """Left arm: magnet A into the tray (red end away from the arms), then keeper 1 across the far end."""
    plan = _pick(L.MAG_A_START[:2], MAG_OPEN, MAG_GRIP, YAW_PICK, "magnet")
    plan += _place(PLACE_A, CARRY, MAG_GRIP, MAG_RELEASE, YAW_PLACE, "magnet")
    plan += _pick(L.KEEP_1_START, OPEN, KEEP_GRIP, 0.0, "keeper", KEEP_PICK_Z)
    plan += _wait(T_B_CLEAR, plan, "hold the keeper until the right hand is back over its keeper")
    plan += _place(PLACE_K_FAR, CARRY_FAR, KEEP_GRIP, KEEP_RELEASE, 0.0, "keeper", z=KEEP_PLACE_Z)
    plan += [Waypoint(1.0, ASIDE_LEFT, KEEP_RELEASE, "aside", 0.0)]
    return plan


def _right() -> list[Waypoint]:
    """Right arm: magnet B into the tray beside A (red end toward the arms), then keeper 2 across the near end."""
    plan = _pick(L.MAG_B_START[:2], MAG_OPEN, MAG_GRIP, YAW_PICK, "magnet")
    plan += _wait(T_A_UP, plan, "hold the magnet until the left hand is up")
    plan += _place(PLACE_B, CARRY, MAG_GRIP, MAG_RELEASE, YAW_PLACE, "magnet")
    plan += _pick(L.KEEP_2_START, OPEN, KEEP_GRIP, 0.0, "keeper", KEEP_PICK_Z)
    plan += _wait(T_K1_UP, plan, "hold the keeper until the left hand is up")
    plan += _place(PLACE_K_NEAR, CARRY, KEEP_GRIP, KEEP_RELEASE, 0.0, "keeper", z=KEEP_PLACE_Z)
    plan += [Waypoint(1.0, ASIDE_RIGHT, KEEP_RELEASE, "aside", 0.0)]
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
