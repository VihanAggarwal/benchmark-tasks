"""On the station table, a butt-welded steel plate (160 x 200 x 10 mm, fixed) lies on the midline about 36 cm in front
of the arm bases, set up for a radiographic test shot with the film under it: its top face is the source side. The
weld (a dark 14 mm cap ground flush) runs along y across its middle at x 0.360, and two yellow ticks across the weld at
y -0.080 and +0.080 mark the ends of the length this shot covers. On the bench in front of the plate lies a wire IQI
(a clear blue 55 x 28 x 12 mm holder with six dark wires along its length, 15 g), wires along y, parallel to the weld;
beside it, one each side, two location markers (22 x 22 x 12 mm lead-figure holders with a white "0" and "1", 15 g).
Rule (ASME Section V, Article 2, T-277.1 and T-275): the IQI on the source side, on the weld with its wires
perpendicular to the weld, inside the shot length; a location marker on the part's source side at each end tick.

The IQI and the markers are loose mechkit compounds (mechkit.py, next to this file): one rigid body each, a holder
and its thin painted wires or figure, so the wires show which way they run and the scorer can read each part's pose
(base frame and orientation). The scorer also reads the arms (TCP, pad-tip point, opening) through rigstate.py, which
wraps the embodiment's step and reset from this file's import, only to check that each part was let go.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the IQI and markers are built as compound bodies)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# The 1.0 state already needs every part released and still for 1 s; success must then hold 1 s more.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "This butt-welded plate is being set up for its radiographic test shot. The film goes under the plate, so the "
    "top of the plate is the side the shot is taken from. The painted stripe is the weld, and the two painted ticks "
    "across it mark the ends of the length this shot covers. Place the wire-type image quality indicator and the "
    "two lead location markers (0 and 1) where ASME Section V requires them for this shot, then let go of them."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


_HY = L.PLATE_HALF_Y
PLATE = [
    _span_box("plate_near", L.PLATE_X0, L.WELD_X0, -_HY, _HY, 0.0, L.PLATE_T, L.STEEL),
    _span_box("weld", L.WELD_X0, L.WELD_X1, -_HY, _HY, 0.0, L.PLATE_T, L.WELD_DARK),
    _span_box("plate_far", L.WELD_X1, L.PLATE_X1, -_HY, _HY, 0.0, L.PLATE_T, L.STEEL),
]
TICKS = [
    _span_box(f"tick_{i}", L.WELD_X - L.TICK_LEN / 2, L.WELD_X + L.TICK_LEN / 2, y - L.TICK_W / 2, y + L.TICK_W / 2,
              L.PLATE_T, L.PLATE_T + L.TICK_T, L.TICK_YELLOW)
    for i, y in enumerate(L.TICK_Y)
]

_WIRES = [mk.box((0.0, y, L.IQI_H + L.WIRE_T / 2), (L.WIRE_LEN, w, L.WIRE_T), color=L.WIRE_DARK)
          for y, w in zip(L.WIRE_Y, L.WIRE_WIDTHS)]
IQI = mk.assembly(
    "iqi",
    origin=(L.IQI_START[0], L.IQI_START[1], L.TOP),
    yaw=math.radians(L.IQI_START[2]),
    mass=L.IQI_MASS,
    color=L.IQI_BLUE,
    shapes=[mk.box((0.0, 0.0, L.IQI_H / 2), (L.IQI_L, L.IQI_W, L.IQI_H)), *_WIRES],
)


def _figure(digit: str) -> list:
    """The white figure on a marker's top, its top toward local +x: '0' a ring 14 x 10 mm, '1' a stroke with a base."""
    z, t, s, h, w = L.MK_H + L.FIG_T / 2, L.FIG_T, L.FIG_STROKE, L.FIG_H, L.FIG_W
    if digit == "0":
        return [mk.box((h / 2 - s / 2, 0.0, z), (s, w, t), color=L.FIG_WHITE),
                mk.box((-h / 2 + s / 2, 0.0, z), (s, w, t), color=L.FIG_WHITE),
                mk.box((0.0, w / 2 - s / 2, z), (h - 2 * s, s, t), color=L.FIG_WHITE),
                mk.box((0.0, -w / 2 + s / 2, z), (h - 2 * s, s, t), color=L.FIG_WHITE)]
    return [mk.box((s / 2, 0.0, z), (h - s, s, t), color=L.FIG_WHITE),  # the stroke
            mk.box((-h / 2 + s / 2, 0.0, z), (s, w * 0.8, t), color=L.FIG_WHITE),  # the base
            mk.box((h / 2 - s, s, z), (s, s, t), color=L.FIG_WHITE)]  # the flag at the top, to its left


def _marker(name: str):
    x, y, yaw = L.MK_START[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP),
        yaw=math.radians(yaw),
        mass=L.MK_MASS,
        color=L.LEAD_GREY,
        shapes=[mk.box((0.0, 0.0, L.MK_H / 2), (L.MK_SIDE, L.MK_SIDE, L.MK_H)), *_figure(name[-1])],
    )


MARKER_0 = _marker("marker_0")
MARKER_1 = _marker("marker_1")
OBJECTS = [*PLATE, *TICKS, IQI, MARKER_0, MARKER_1]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and a teleport back to the start are
    two more signals)."""
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


def _parts(objects) -> dict:
    return {n: objects[n] for n in L.NAMES}


def _pose(s) -> dict:
    """A part's base-frame origin (mm), heading (deg) and source, for the trace."""
    try:
        r = s.quat_xyzw
        return {"origin_mm": [round(float(v) * 1000, 1) for v in s.origin], "quat": [round(float(v), 5) for v in r],
                "source": getattr(s, "source", None)}
    except (AttributeError, TypeError, ValueError):
        return {"origin_mm": None}


def success(objects) -> bool:
    """The IQI flat on the plate's top, its wires within 8 degrees of perpendicular to the weld, its centre within
    15 mm of the weld centreline and its footprint between the ticks; a location marker flat on the plate's top within
    10 mm of each tick and 35 mm of the weld centreline; all three released and still for 1 s; held 1 s. This runs
    first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    parts = _parts(objects)
    result = _TRACKER.update(parts, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    row.update({f"{n}_pose": _pose(parts[n]) for n in L.NAMES})
    if isinstance(arms, dict):
        for side in ("left", "right"):
            a = arms.get(side)
            row[f"{side}_tips"] = getattr(a, "tips", None)
            row[f"{side}_opening"] = getattr(a, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0: IQI correct (flat on the plate's top, wires within 8 degrees of perpendicular to the weld, centre within
    15 mm of the weld centreline, footprint between the ticks), a correct marker at both ticks, all still for 1 s;
    0.75: IQI correct and a correct marker at one tick or more; 0.5: IQI correct, no correct marker; 0.25: IQI lying
    flat on the plate's top but not correct; else 0. A part counts only released (no finger zone within 10 mm) and
    settled (under 2 mm and 1 degree over 0.5 s). Read from the state at hand (the last step)."""
    return _TRACKER.result(_parts(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
IQI_OPEN = -0.022  # 44 mm: 8 mm either side of the IQI's 28 mm width
IQI_GRIP = -(L.IQI_W - 0.008) / 2  # 20 mm commanded across 28 mm
MK_OPEN = -0.020  # 40 mm: 9 mm either side of a 22 mm marker
MK_GRIP = -(L.MK_SIDE - 0.008) / 2  # 14 mm commanded across 22 mm
PAD = 0.013  # the pad tips are this far below the tool point
GRIP_UP = 0.003 + PAD  # the TCP 16 mm above a part's underside: pad tips 3 mm above it, holding 9 of its 12 mm
DROP = 0.001  # each part is let go with its underside 1 mm above the plate's top
APPROACH = 0.030  # each final descent starts this far above its end point and takes 1.5 s
HOVER = L.TOP + 0.055  # carried with a part's underside 39 mm up: 29 mm over the plate, 16 mm over a laid IQI
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.PLATE_TOP_W + DROP + GRIP_UP
# Yaw 0 closes the fingers along y. The IQI starts at yaw +90 (wires along y): gripped across its 28 mm width at yaw
# +90 (fingers closing along x), carried while the gripper turns a quarter turn clockwise to yaw 0, it is laid at yaw
# 0, wires along x. The markers are gripped and laid at yaw 0.
# Place-yaw trim, for the GPU run: in our runs parts carried in a gripper ended +0.6 to +1.7 degrees counter-clockwise
# of the commanded yaw, so the IQI is laid at -1.2 degrees (8 degrees allowed). If iqi_wire_err_deg in TASK_TRACE
# after 'let go of the IQI' is over 3 degrees, change YAW_TRIM by minus the IQI's heading.
YAW_TRIM = math.radians(-1.2)
IQI_YAW_PICK = math.radians(L.IQI_START[2])
IQI_YAW_SET = math.radians(L.IQI_TARGET_YAW) + YAW_TRIM
MK_YAW = 0.0
# Place trim, for the GPU run: parts have landed about 4 mm short of the place point (toward the arm's base), so
# each place point is moved 3 mm outward along the line from the placing arm's base. If a part lands off its target
# by more than 3 mm (iqi_off_weld_mm away from 0, marker_*_from_weld_mm away from -28 or marker_*_y_mm away from
# -+80), change LAND_SHORT or add a PLACE_TRIM offset of minus that amount.
LAND_SHORT = 0.003
PLACE_TRIM = {"iqi": (0.0, 0.0), "marker_0": (0.0, 0.0), "marker_1": (0.0, 0.0)}
ARM_OF = {"iqi": "right", "marker_0": "right", "marker_1": "left"}
# Where each arm waits and ends: high over its own side of the bench, clear of the plate for the top camera.
STANDBY = {"left": (0.200, 0.300, L.TOP + 0.150), "right": (0.200, -0.300, L.TOP + 0.150)}


def place_point(name: str) -> tuple[float, float]:
    tx, ty = L.IQI_TARGET if name == "iqi" else L.MK_TARGET[name]
    bx, by = L.BASES[ARM_OF[name]]
    r = math.hypot(tx - bx, ty - by)
    dx, dy = PLACE_TRIM[name]
    return tx + LAND_SHORT * (tx - bx) / r + dx, ty + LAND_SHORT * (ty - by) / r + dy


def start_point(name: str) -> tuple[float, float]:
    x, y, _ = L.IQI_START if name == "iqi" else L.MK_START[name]
    return x, y


def _pick(name: str, opn: float, grip: float, yaw: float, first: float) -> list[Waypoint]:
    """Over the part, down (slowly for the last 30 mm), grip it across two faces, lift."""
    sx, sy = start_point(name)
    return [
        Waypoint(first, (sx, sy, HOVER), opn, f"over {name}", yaw),
        Waypoint(1.0, (sx, sy, PICK_Z + APPROACH), opn, f"down to just above {name}", yaw),
        Waypoint(1.5, (sx, sy, PICK_Z), opn, f"slowly down round {name}", yaw),
        Waypoint(0.5, (sx, sy, PICK_Z), opn, "settle", yaw),
        Waypoint(1.5, (sx, sy, PICK_Z), grip, f"grip {name}", yaw),
        Waypoint(0.3, (sx, sy, PICK_Z), grip, "settle the grip", yaw),
        Waypoint(1.2, (sx, sy, HOVER), grip, f"lift {name}", yaw),
    ]


def _place(name: str, opn: float, grip: float, yaw: float, carry: float, what: str) -> list[Waypoint]:
    """Carry it over its place, down (slowly), let go 1 mm above the plate, straight up."""
    px, py = place_point(name)
    return [
        Waypoint(carry, (px, py, HOVER), grip, f"carry {name} {what}", yaw),
        Waypoint(0.5, (px, py, HOVER), grip, "settle over the plate", yaw),
        Waypoint(1.0, (px, py, PLACE_Z + APPROACH), grip, "down to just above the plate", yaw),
        Waypoint(1.5, (px, py, PLACE_Z), grip, f"slowly lay {name} on the plate", yaw),
        Waypoint(0.5, (px, py, PLACE_Z), grip, "settle", yaw),
        Waypoint(1.5, (px, py, PLACE_Z), opn, f"let go of {name}", yaw),
        Waypoint(1.0, (px, py, HOVER), opn, "straight up, clear of it", yaw),
    ]


def _right() -> list[Waypoint]:
    """The IQI (0 to 16.5 s), then marker 0: picked by 24.5 s, held up over its start until the left arm has laid
    marker 1 and gone up (32.5 s), laid at the -y tick, then the arm goes back over its side of the bench (42 s)."""
    sx, sy = start_point("marker_0")
    return [
        *_pick("iqi", IQI_OPEN, IQI_GRIP, IQI_YAW_PICK, 2.0),
        *_place("iqi", IQI_OPEN, IQI_GRIP, IQI_YAW_SET, 2.5, "over the weld, turning its wires across it"),
        *_pick("marker_0", MK_OPEN, MK_GRIP, MK_YAW, 2.0),
        Waypoint(8.0, (sx, sy, HOVER), MK_GRIP, "hold it up while the left arm lays marker_1", MK_YAW),
        *_place("marker_0", MK_OPEN, MK_GRIP, MK_YAW, 2.0, "to the -y tick, beside the weld"),
        Waypoint(1.5, STANDBY["right"], MK_OPEN, "back over the bench", MK_YAW),
    ]


def _left() -> list[Waypoint]:
    """Waits high over its side of the bench until the right arm has laid the IQI and moved off to marker 0 (16.5 s),
    then marker 1 (picked by 24.5 s, laid at the +y tick and the gripper up by 32.5 s), then back (34 s)."""
    return [
        Waypoint(2.0, STANDBY["left"], MK_OPEN, "wait over the bench", MK_YAW),
        Waypoint(14.5, STANDBY["left"], MK_OPEN, "wait while the right arm lays the IQI and goes", MK_YAW),
        *_pick("marker_1", MK_OPEN, MK_GRIP, MK_YAW, 2.0),
        *_place("marker_1", MK_OPEN, MK_GRIP, MK_YAW, 2.0, "to the +y tick, beside the weld"),
        Waypoint(1.5, STANDBY["left"], MK_OPEN, "back over the bench", MK_YAW),
    ]


def demo():
    """Both arms on one clock: the right arm lays the IQI and then marker 0, the left arm marker 1 in between, so the
    two grippers are never both near the plate. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling
    check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
