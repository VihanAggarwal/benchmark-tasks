"""On the station table, a bench mock-up of a split air conditioner's condensing-unit service corner, all fixed: the
unit's grey corner (12 x 6 x 16 cm) 23 to 35 cm in front of the arm bases, 22 to 28 cm left of the midline; on its
right face two brass service valves; from them two copper lines run to the robot's right along y: the 19 mm suction
line (axis 25 cm out, 7 cm up: a brazed coupling 2.5 cm long at the valve, 24.5 cm of bare copper, then 7 cm under
black foam insulation) and the 9.5 mm liquid line (32 cm out). Loose: a blue pipe-clamp thermometer (a 60 g saddle,
its legs 33 mm apart inside) on a grey pad, and a black foam cover (a 20 g half-shell with a 65 mm channel) on
another. The clamp goes astride the suction line on bare copper, its middle within 15 cm of the suction valve's body,
the cover over it, and both grippers let go.

The clamp and the cover are mechkit assemblies (mechkit.py, next to this file: each one loose rigid body made of
several boxes); the scorer reads their poses from it, and the arms (tool point, pad tips) through rigstate.py, which
wraps the embodiment's step and reset from this file's import. Neither changes what the embodiment does.
offline_saddle.py is used only by the offline kinematic check (no physics there): the simulator never calls it.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the two parts are built as compound bodies)
import offline_saddle  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Held for 1 s, which can start only once both grippers are clear of both parts (tool point and pad tips 4 cm above
# them or 6.5 cm to the side): the run cannot end early while a gripper still holds or touches either.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Mount the blue pipe-clamp thermometer on the suction line: set it astride the line, on bare copper with its "
    "middle within 15 cm of the suction service valve's body and clear of the brazed joint next to the valve. Then "
    "set the black foam cover over the clamp to insulate it, and let go of both."
)


def _box(name: str, centre, size, color) -> Box:
    return Box(name, pos=(centre[0], centre[1], L.TOP + centre[2]), size=size, color=color)


def _cap(name: str, centre, radius: float, height: float) -> Cylinder:
    return Cylinder(name, pos=(centre[0], centre[1], L.TOP + centre[2]), radius=radius, height=height, color=L.BRASS)


RIG = [
    _box("unit_corner", L.UNIT_C, L.UNIT_S, L.STEEL_GREY),
    _box("suction_valve", L.SVALVE_C, L.SVALVE_S, L.BRASS),
    _cap("suction_valve_cap", L.SVALVE_CAP_C, L.SVALVE_CAP_R, L.SVALVE_CAP_H),
    _box("liquid_valve", L.LVALVE_C, L.LVALVE_S, L.BRASS),
    _cap("liquid_valve_cap", L.LVALVE_CAP_C, L.LVALVE_CAP_R, L.LVALVE_CAP_H),
    _box("brazed_coupling", L.COUPLING_C, L.COUPLING_S, L.COUPLING_COPPER),
    _box("suction_line", L.BARE_C, L.BARE_S, L.COPPER),
    _box("suction_foam", L.FOAM_C, L.FOAM_S, L.FOAM_BLACK),
    _box("liquid_line", L.LIQ_C, L.LIQ_S, L.COPPER),
    _box("foam_stand", L.FOAM_STAND_C, L.FOAM_STAND_S, L.STAND_GREY),
    _box("liquid_stand", L.LIQ_STAND_C, L.LIQ_STAND_S, L.STAND_GREY),
    _box("clamp_pad", L.CLAMP_PAD_C, L.CLAMP_PAD_S, L.PAD_GREY),
    _box("cover_pad", L.COVER_PAD_C, L.COVER_PAD_S, L.PAD_GREY),
]
CLAMP = mk.assembly(
    "clamp", origin=(L.CLAMP_START[0], L.CLAMP_START[1], L.TOP + L.CLAMP_START[2]), mass=L.CLAMP_MASS,
    color=L.CLAMP_BLUE, shapes=[mk.box(c, s, color=col) for c, s, col in L.CLAMP_SHAPES],
)
COVER = mk.assembly(
    "cover", origin=(L.COVER_START[0], L.COVER_START[1], L.TOP + L.COVER_START[2]), mass=L.COVER_MASS,
    color=L.COVER_BLACK, shapes=[mk.box(c, s, color=col) for c, s, col in L.COVER_SHAPES],
)
OBJECTS = [*RIG, CLAMP, COVER]
SPECS = {"clamp": CLAMP, "cover": COVER}

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
    """The clamp and the cover the scorer judges this step, and whether they come from the offline checker. In the
    simulator: the parts' states as mechkit reads them. Offline (the kinematic check: no physics, and its states say
    source "offline") offline_saddle moves them to where they really rest astride (the checker sees each
    part as its box). The simulator never takes that branch (its states say source "sim")."""
    clamp, cover = objects.get("clamp"), objects.get("cover")
    offline = "offline" in (getattr(clamp, "source", None), getattr(cover, "source", None))
    info = {}
    if offline:
        clamp, cover, info = offline_saddle.correct(clamp, cover, SPECS)
    return clamp, cover, offline, info


def success(objects) -> bool:
    """The clamp seated astride the suction line on bare copper with d <= 15 cm, the cover over it with its ceiling on
    the clamp's display, both at rest, and both grippers clear of both parts; held 1 s. This runs first every control
    step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    clamp, cover, offline, info = _observe(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(clamp, cover, arms, offline)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "source": getattr(clamp, "source", None), **{f"offline_{k}": v for k, v in info.items()}}
    try:
        row.update(clamp_origin=clamp.origin, clamp_quat=clamp.quat_xyzw, cover_origin=cover.origin,
                   cover_quat=cover.quat_xyzw)
    except AttributeError:
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips, opening={s: arms[s].opening for s in ("left", "right")})
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 for the clamp seated astride the suction line on bare copper with d (valve face to clamp centre) <= 15 cm,
    the cover over it resting on its display, and both grippers clear of both parts. Otherwise, with both grippers
    clear of the clamp: 0.5 for the clamp so seated with d <= 15 cm; 0.25 for it seated with d over 15 cm. 0 in
    every other case. Clear of a part: a gripper's tool point and pad tips 4 cm above its box or 6.5 cm to the side."""
    clamp, cover, offline, _ = _observe(objects)
    return _TRACKER.result(clamp, cover, rigstate.arms(), offline).score


# -- the scripted demo (left arm; the right arm holds its ready pose) -------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
CLAMP_GRIP = -(L.BLOCK_S[1] - 0.008) / 2  # yaw 0: the fingers close along y, across the top block's 40 mm
COVER_GRIP = -(L.COVER_W - 0.008) / 2  # yaw -90 deg: the fingers close along x, across the cover's 77 mm walls
YAW_CLAMP = 0.0
YAW_COVER = -math.pi / 2  # the left arm's sign for fingers along x
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
CLAMP_TCP_UP = 0.017  # the TCP this far above the clamp's origin: pad tips 4 mm above the block's underside (12 mm
#                       of the 16 mm block held); the block's underside meets the copper 4 mm before the tips would
COVER_TCP_UP = 0.030  # the TCP this far above the cover's bottom: pad tips 17 mm up its 36 mm walls
TRAVEL = 0.150  # the clamp is carried with its legs' bottom 11 cm above the table
COVER_TRAVEL = 0.170  # the cover is carried with its bottom 14 cm above the table, clear of the seated clamp
FINAL_Z = 0.180  # the last rise: the pad tips 5.75 cm above the cover's top (clear needs 4 cm)
PLACE_XY = (L.LINE_X, 0.090)  # d = 100 mm: 5 cm inside the 15 cm limit and 5.5 cm clear of the coupling's end
CLAMP_PLACE_Z = L.LINE_TOP + 0.002 + CLAMP_TCP_UP  # let go with the underside 2 mm above the copper
CLAMP_ENTRY_Z = CLAMP_PLACE_Z + 0.030  # the legs' bottom 12 mm above the copper: the last 30 mm slowly
COVER_PLACE_Z = L.LINE_TOP + L.DISPLAY_TOP + 0.002 - L.CEILING + COVER_TCP_UP  # ceiling 2 mm above the display
COVER_ENTRY_Z = COVER_PLACE_Z + 0.040  # the walls' bottom 6 mm above the display's top: the last 40 mm slowly
# GPU trims (x, y) in metres, added to the place points. The legs leave 7 mm a side over the copper and the walls
# 8 mm a side over the clamp, but a loaded arm trails its command by 3-7 mm. If the trace shows the clamp landing off
# the line's axis or off d = 100 mm, set CLAMP_TRIM to (-clamp_x_off_mm, d_mm - 100) / 1000, and COVER_TRIM to minus
# the cover's own offset from the planned point (cover_off_mm plus the clamp's (clamp_x_off_mm, 100 - d_mm), over
# 1000), so that both land on it.
CLAMP_TRIM = (0.0, 0.0)
COVER_TRIM = (0.0, 0.0)


def _left() -> list[Waypoint]:
    """Left arm: pick the clamp off its pad by its top block, set it astride the suction line 10 cm from the valve,
    let go; pick the cover by its walls, set it down over the clamp, let go, rise."""
    cx, cy = L.CLAMP_START[0], L.CLAMP_START[1]
    vx, vy = L.COVER_START[0], L.COVER_START[1]
    px, py = PLACE_XY[0] + CLAMP_TRIM[0], PLACE_XY[1] + CLAMP_TRIM[1]
    qx, qy = PLACE_XY[0] + COVER_TRIM[0], PLACE_XY[1] + COVER_TRIM[1]
    pick_c = L.TOP + L.CLAMP_START[2] + CLAMP_TCP_UP
    pick_v = L.TOP + L.COVER_START[2] + COVER_TCP_UP
    z = lambda h: L.TOP + h  # noqa: E731
    yc, yv = YAW_CLAMP, YAW_COVER
    return [
        Waypoint(3.0, (cx, cy, z(TRAVEL)), OPEN, "over the clamp", yc),
        Waypoint(1.0, (cx, cy, pick_c + 0.030), OPEN, "down beside it", yc),
        Waypoint(1.5, (cx, cy, pick_c), OPEN, "the last 3 cm slowly", yc),
        Waypoint(0.6, (cx, cy, pick_c), OPEN, "settle", yc),
        Waypoint(1.5, (cx, cy, pick_c), CLAMP_GRIP, "grip the clamp's top block", yc),
        Waypoint(2.0, (cx, cy, z(TRAVEL)), CLAMP_GRIP, "lift it", yc),
        Waypoint(3.0, (px, py, z(TRAVEL)), CLAMP_GRIP, "carry it over the bare suction line", yc),
        Waypoint(1.0, (px, py, z(CLAMP_ENTRY_Z)), CLAMP_GRIP, "down to just above the line", yc),
        Waypoint(0.5, (px, py, z(CLAMP_ENTRY_Z)), CLAMP_GRIP, "settle over the line", yc),
        Waypoint(2.0, (px, py, z(CLAMP_PLACE_Z)), CLAMP_GRIP, "lower it astride the line slowly", yc),
        Waypoint(0.6, (px, py, z(CLAMP_PLACE_Z)), CLAMP_GRIP, "settle", yc),
        Waypoint(1.5, (px, py, z(CLAMP_PLACE_Z)), OPEN, "let go of the clamp", yc),
        Waypoint(2.0, (px, py, z(TRAVEL)), OPEN, "straight up", yc),
        Waypoint(3.0, (vx, vy, z(TRAVEL)), OPEN, "over the cover, turning the fingers along x", yv),
        Waypoint(1.0, (vx, vy, pick_v + 0.030), OPEN, "down beside it", yv),
        Waypoint(1.5, (vx, vy, pick_v), OPEN, "the last 3 cm slowly", yv),
        Waypoint(0.6, (vx, vy, pick_v), OPEN, "settle", yv),
        Waypoint(1.5, (vx, vy, pick_v), COVER_GRIP, "grip the cover by its walls", yv),
        Waypoint(2.0, (vx, vy, z(COVER_TRAVEL)), COVER_GRIP, "lift it", yv),
        Waypoint(2.0, (qx, qy, z(COVER_TRAVEL)), COVER_GRIP, "carry it over the clamp", yv),
        Waypoint(1.0, (qx, qy, z(COVER_ENTRY_Z)), COVER_GRIP, "down to just above the clamp", yv),
        Waypoint(0.5, (qx, qy, z(COVER_ENTRY_Z)), COVER_GRIP, "settle over the clamp", yv),
        Waypoint(2.5, (qx, qy, z(COVER_PLACE_Z)), COVER_GRIP, "lower it over the clamp slowly", yv),
        Waypoint(0.6, (qx, qy, z(COVER_PLACE_Z)), COVER_GRIP, "settle", yv),
        Waypoint(1.5, (qx, qy, z(COVER_PLACE_Z)), OPEN, "let go of the cover", yv),
        Waypoint(2.0, (qx, qy, z(FINAL_Z)), OPEN, "straight up, clear of both", yv),
    ]


def demo():
    """One arm: the right arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
