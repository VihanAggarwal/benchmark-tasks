"""On the station table: a horizontal gel electrophoresis tank of clear acrylic on the midline about 32 cm in front of
the arm bases (outside 106 x 240 mm, walls 45 mm tall, its two side walls' inner faces 94 mm apart), with a grey
casting platform in its middle and its two electrodes at the ends (black at the right end, red at the left). On the
bench on the left lies a UV-clear gel tray, open at both ends with a black rubber gasket on each open end (84 mm
gasket to gasket, 64 mm wide, 18 mm tall, three comb slots across it), lengthwise (its length along y). On the right a
black comb (a 72 mm bar with its teeth below the bar's middle) stands in a white comb stand. Rule: to cast a gel in
this tank the tray goes onto the platform turned crosswise, so its gasketed open ends face the side walls and are
sealed by them, and the comb stands in the slot nearest one end of the tray; then let go of both.

The tray and the comb are loose mechkit compounds (mechkit.py, next to this file): each one rigid body of several
boxes, so the tray has real side walls, notches and gaskets and the comb real teeth, and the scorer can read their
poses (base frame and orientation). The scorer also reads the arms (TCP, pad-tip point, opening) through rigstate.py,
which wraps the embodiment's step and reset from this file's import; neither changes anything else the embodiment
does. offline_nest.py is used only by the offline kinematic check (no physics there): the simulator never calls it.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the tray and comb are built as compound bodies)
import offline_nest  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Held 1 s. Every tier needs both grippers' finger zones at least 20 mm from the tray, and the comb counts only with
# them 20 mm from it too, so the trial cannot end while a gripper still holds or touches either part.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Set the gel box up to cast a gel. Put the gel tray on the grey platform in the tank turned crosswise, so its two "
    "open ends, which carry the black gaskets, face the tank's two side walls and are sealed by them (not toward the "
    "electrodes). Then stand the comb in the comb slot nearest one end of the tray, across the tray with its bar down "
    "in the notches of both side walls. Let go of both."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


_HY, _EY, (_SX0, _SX1) = L.TANK_HALF_Y, L.END_IN, L.SIDE_IN
TANK = [
    _span_box("tank_side_near", L.TANK_X0, _SX0, -_HY, _HY, 0.0, L.TANK_H, L.CLEAR_ACRYLIC),
    _span_box("tank_side_far", _SX1, L.TANK_X1, -_HY, _HY, 0.0, L.TANK_H, L.CLEAR_ACRYLIC),
    _span_box("tank_end_left", _SX0, _SX1, _EY, _HY, 0.0, L.TANK_H, L.CLEAR_ACRYLIC),
    _span_box("tank_end_right", _SX0, _SX1, -_HY, -_EY, 0.0, L.TANK_H, L.CLEAR_ACRYLIC),
    _span_box("tank_floor", _SX0, _SX1, -_EY, _EY, 0.0, L.TANK_FLOOR, L.CLEAR_ACRYLIC),
]
PLATFORM = _span_box("casting_platform", L.PLAT_X0, L.PLAT_X1, -L.PLAT_HALF_Y, L.PLAT_HALF_Y, L.TANK_FLOOR,
                     L.PLAT_TOP, L.PLATFORM_GREY)
_ex, _ey, _ez = L.ELECTRODE_SIZE
ELECTRODES = [
    _span_box("electrode_black", L.TANK_X - _ex / 2, L.TANK_X + _ex / 2, -L.ELECTRODE_Y - _ey / 2,
              -L.ELECTRODE_Y + _ey / 2, L.TANK_FLOOR, L.TANK_FLOOR + _ez, L.CATHODE_BLACK),
    _span_box("electrode_red", L.TANK_X - _ex / 2, L.TANK_X + _ex / 2, L.ELECTRODE_Y - _ey / 2,
              L.ELECTRODE_Y + _ey / 2, L.TANK_FLOOR, L.TANK_FLOOR + _ez, L.ANODE_RED),
]
STAND = [
    Box(f"comb_stand_{side}", pos=(x, L.STAND_Y, L.TOP + L.STAND_PLATE[2] / 2), size=L.STAND_PLATE,
        color=L.STAND_WHITE)
    for side, x in zip(("near", "far"), L.STAND_PLATE_X)
]


def _tray_shapes() -> list:
    """The tray in its own frame (origin at the middle of its underside, x along its length): a floor, two side walls
    with three notches in each top (the comb slots), and a black gasket on each open end."""
    half = L.TRAY_BODY_LEN / 2
    wall_z0 = L.TRAY_FLOOR
    shapes = [mk.box((0.0, 0.0, L.TRAY_FLOOR / 2), (L.TRAY_BODY_LEN, L.TRAY_W, L.TRAY_FLOOR))]
    # the solid stretches of each wall's top 5 mm, between and beyond the three notches
    edges = sorted([-half, half] + [c + s * L.NOTCH_W / 2 for c in L.SLOTS.values() for s in (-1, 1)])
    tops = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    for sy in (-1.0, 1.0):
        y = sy * L.WALL_Y
        shapes.append(mk.box((0.0, y, (wall_z0 + L.NOTCH_BOTTOM) / 2),
                             (L.TRAY_BODY_LEN, L.TRAY_WALL, L.NOTCH_BOTTOM - wall_z0)))
        for x0, x1 in tops:
            shapes.append(mk.box(((x0 + x1) / 2, y, (L.NOTCH_BOTTOM + L.TRAY_H) / 2),
                                 (x1 - x0, L.TRAY_WALL, L.TRAY_H - L.NOTCH_BOTTOM)))
    for sx in (-1.0, 1.0):
        shapes.append(mk.box((sx * (half + L.GASKET_SIZE[0] / 2), 0.0, L.GASKET_SIZE[2] / 2), L.GASKET_SIZE,
                             color=L.GASKET_BLACK))
    return shapes


TRAY = mk.assembly(
    "gel_tray",
    origin=(L.TRAY_START[0], L.TRAY_START[1], L.TOP),
    yaw=L.TRAY_START_YAW,
    mass=L.TRAY_MASS,
    color=L.TRAY_CLEAR,
    shapes=_tray_shapes(),
)
COMB = mk.assembly(
    "comb",
    origin=(L.COMB_START[0], L.COMB_START[1], L.TOP + L.COMB_START[2]),
    mass=L.COMB_MASS,
    color=L.COMB_BLACK,
    shapes=[
        mk.box((0.0, 0.0, L.COMB_BAR[2] / 2), L.COMB_BAR),
        mk.box((0.0, 0.0, -L.COMB_TEETH[2] / 2), L.COMB_TEETH),
    ],
)
OBJECTS = [*TANK, PLATFORM, *ELECTRODES, *STAND, TRAY, COMB]

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


def _observe(objects):
    """The tray and the comb the scorer judges this step. In the simulator: the parts' states as mechkit reads them.
    Offline (the kinematic check: no physics, and its states say source "offline") offline_nest moves the comb to where
    it really rests in the tray (the checker sees each part as its box). The simulator never takes that branch."""
    tray, comb = objects.get("gel_tray"), objects.get("comb")
    info = {}
    if "offline" in (getattr(tray, "source", None), getattr(comb, "source", None)):
        comb, info = offline_nest.correct(tray, comb, COMB)
    return tray, comb, info


def success(objects) -> bool:
    """The tray flat on the platform, crosswise between the side walls (each gasketed end within 10 mm of its wall),
    the comb seated in an end slot, both at rest, both grippers' finger zones 20 mm or more from both; held 1 s. This
    runs first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    tray, comb, info = _observe(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(tray, comb, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details,
           "source": getattr(tray, "source", None), **{f"offline_{k}": v for k, v in info.items()}}
    try:
        row.update(tray_origin=tray.origin, tray_quat=tray.quat_xyzw, comb_origin=comb.origin,
                   comb_quat=comb.quat_xyzw)
    except AttributeError:
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips, openings=[arms["left"].opening, arms["right"].opening])
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the tray is set to cast (flat on the platform: tilted 3 degrees or less, every bottom corner within
    2 mm of its top and over the platform; crosswise: its length within 12 degrees of the line between the side walls;
    each gasketed end within 10 mm of its side wall), at rest, with both grippers' finger zones 20 mm or more from it. Then 1.0 if the
    comb is seated in a slot near an end (bar down in the notches of both side walls), at rest, both grippers 20 mm or
    more from it; 0.75 if seated so in the middle slot; 0.5 otherwise. Read from the state at hand (the last step)."""
    tray, comb, _ = _observe(objects)
    return _TRACKER.result(tray, comb, rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
TRAY_GRIP = -(L.TRAY_W - 0.008) / 2  # across the tray's 64 mm width, on its side walls: 56 mm commanded
COMB_GRIP = -(L.COMB_BAR[1] - 0.008) / 2  # on the ends of the comb's 72 mm bar: 64 mm commanded
# The tray lies lengthwise (length along y): the left arm grips it across its width with the fingers closing along x
# (yaw -pi/2) and turns the grip to yaw 0 while carrying it, so it arrives crosswise (length along x) with the
# fingers closing along y on its side walls, 33.5 mm clear of the tank's side walls. The comb stands with its bar along
# y: the right arm grips the bar's ends (yaw 0) and never turns it.
YAW_TRAY_PICK, YAW_TRAY_PLACE, YAW_COMB = -math.pi / 2, 0.0, 0.0
TRAY_GRIP_UP = 0.018  # the TCP 18 mm above the tray's underside: pad tips 5 mm above it, on its walls' outer faces
COMB_GRIP_UP = 0.021  # the TCP 21 mm above the bar's underside: pad tips 8 mm above it, on the bar's top 10 mm
HOVER = L.TOP + 0.13
HOVER_TANK = L.TOP + 0.11  # over the tank: lower, well inside both arms' reach over the midline (0.13 there leaves
# joint 4 only 4-6 degrees from its limit); the carried tray's underside is still 47 mm above the tank's walls
DROP = 0.0005  # each part is let go this far above where it rests
TRAY_PICK = (L.TRAY_START[0], L.TRAY_START[1], L.TOP + TRAY_GRIP_UP)
TRAY_PLACE = (L.TANK_X, L.TANK_Y, L.TOP + L.PLAT_TOP + DROP + TRAY_GRIP_UP)  # 34.5 mm up: pad tips 21.5 mm up
TRAY_ENTRY = L.TOP + L.TANK_H + 0.0065 + TRAY_GRIP_UP  # the tray's underside 6.5 mm above the tank's walls
COMB_PICK = (L.COMB_START[0], L.COMB_START[1], L.TOP + L.COMB_START[2] + COMB_GRIP_UP)
# The slot nearer the arms: 25 mm toward them from the tray's middle (the tray is the same either way round). Its
# bar comes down with its pad tips 37.5 mm up, above the tray's walls (34 mm), the fingers 8.5 mm from the near side
# wall's inner face.
SLOT_X = L.TANK_X - L.SLOTS["end_plus"]
COMB_PLACE = (SLOT_X, L.TANK_Y, L.TOP + L.PLAT_TOP + L.NOTCH_BOTTOM + DROP + COMB_GRIP_UP)  # 50.5 mm up
COMB_ENTRY = COMB_PLACE[2] + 0.025  # the teeth 13.5 mm above the tray's walls (2.5 mm above the tank's)
APPROACH = 0.030  # the final descent at each pick starts this far above
LEFT_PARK = (0.22, 0.22, L.TOP + 0.15)
RIGHT_WAIT = (0.20, -0.24, HOVER)
RIGHT_PARK = (0.22, -0.20, L.TOP + 0.15)
# Payload sag: the arms' controller does not carry a part's weight. The compliance model (J K^-1 J^T of the part's
# weight, joints 80/80/80/10/10/10 N m/rad, yamkin) puts the left gripper about (-0.4, +0.4, -1.2) mm off at the
# tray's place pose under the 38 g tray (the tray meets the platform about 0.7 mm before the slow descent ends) and the
# right one about (-0.2, -0.2, -0.6) mm off at the slot under the 20 g comb: far inside the 5 mm of play at the side
# walls and in the slot, so no trim. On the GPU read gasket_gap_mm and comb_seat_x_mm in TASK_TRACE: if the comb's
# bar sits more than 2 mm off the slot's centre (|comb_seat_x_mm| should read 25; the sign only says which way round
# the tray is), set SLOT_X to the tray's x where it came to rest (tray_origin) minus 25 mm.


def _left() -> list[Waypoint]:
    """Left arm: grip the tray across its width where it lies lengthwise, lift it, turn it a quarter turn while
    carrying it over the tank, lower it between the side walls onto the platform, let go, go up and aside."""
    px, py, pz = TRAY_PICK
    tx, ty, tz = TRAY_PLACE
    a, b = YAW_TRAY_PICK, YAW_TRAY_PLACE
    return [
        Waypoint(2.0, (px, py, HOVER), OPEN, "over the tray", a),
        Waypoint(1.5, (px, py, pz + APPROACH), OPEN, "down to just above it", a),
        Waypoint(1.5, (px, py, pz), OPEN, "slowly down round its side walls", a),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", a),
        Waypoint(1.5, (px, py, pz), TRAY_GRIP, "grip the tray", a),
        Waypoint(0.5, (px, py, pz), TRAY_GRIP, "settle the grip", a),
        Waypoint(1.5, (px, py, HOVER), TRAY_GRIP, "lift it", a),
        Waypoint(2.5, (tx, ty, HOVER_TANK), TRAY_GRIP, "carry it over the tank, turning it crosswise", b),
        Waypoint(0.8, (tx, ty, HOVER_TANK), TRAY_GRIP, "settle over the platform", b),
        Waypoint(1.0, (tx, ty, TRAY_ENTRY), TRAY_GRIP, "down to just above the walls", b),
        Waypoint(2.0, (tx, ty, tz), TRAY_GRIP, "slowly down between the side walls onto the platform", b),
        Waypoint(0.5, (tx, ty, tz), TRAY_GRIP, "settle", b),
        Waypoint(1.5, (tx, ty, tz), OPEN, "let go of the tray", b),
        Waypoint(1.0, (tx, ty, HOVER_TANK), OPEN, "straight up", b),
        Waypoint(1.5, LEFT_PARK, OPEN, "aside", b),
    ]


def _right() -> list[Waypoint]:
    """Right arm: grip the comb's bar by its ends, lift it out of its stand, wait aside until the tray is in and the
    left arm has gone, carry it over the slot nearer the arms, lower it into the notches, let go, go up and aside."""
    px, py, pz = COMB_PICK
    sx, sy, sz = COMB_PLACE
    yaw = YAW_COMB
    first = [
        Waypoint(2.0, (px, py, HOVER), OPEN, "over the comb", yaw),
        Waypoint(1.5, (px, py, pz + APPROACH), OPEN, "down to just above it", yaw),
        Waypoint(1.5, (px, py, pz), OPEN, "slowly down round the bar's ends", yaw),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", yaw),
        Waypoint(1.5, (px, py, pz), COMB_GRIP, "grip the comb", yaw),
        Waypoint(0.5, (px, py, pz), COMB_GRIP, "settle the grip", yaw),
        Waypoint(1.5, (px, py, HOVER), COMB_GRIP, "lift it straight out of the stand", yaw),
        Waypoint(1.5, RIGHT_WAIT, COMB_GRIP, "aside, to wait", yaw),
    ]
    # wait until the left arm has let go of the tray, gone up and is halfway to its park
    left = _left()
    go = sum(w.duration for w in left[:-1]) + 0.7
    wait = go - sum(w.duration for w in first)
    return first + [
        Waypoint(wait, RIGHT_WAIT, COMB_GRIP, "wait for the tray and the left arm", yaw),
        Waypoint(2.0, (sx, sy, HOVER_TANK), COMB_GRIP, "over the slot nearer the arms", yaw),
        Waypoint(0.8, (sx, sy, HOVER_TANK), COMB_GRIP, "settle over the slot", yaw),
        Waypoint(1.0, (sx, sy, COMB_ENTRY), COMB_GRIP, "down to just above the tray", yaw),
        Waypoint(2.0, (sx, sy, sz), COMB_GRIP, "slowly down into the notches", yaw),
        Waypoint(0.5, (sx, sy, sz), COMB_GRIP, "settle", yaw),
        Waypoint(1.5, (sx, sy, sz), OPEN, "let go of the comb", yaw),
        Waypoint(1.0, (sx, sy, HOVER_TANK), OPEN, "straight up", yaw),
        Waypoint(1.5, RIGHT_PARK, OPEN, "aside", yaw),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
