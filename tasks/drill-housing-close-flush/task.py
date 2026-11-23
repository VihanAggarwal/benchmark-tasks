"""On the station table, fixed in a three-block nest: the open left half of a compact 20 V cordless drill/driver's
clamshell housing (orange, 131 mm barrel along x, a 100 mm handle toward the right ending in a 120 x 30 mm battery
foot), open side up, its parting plane 28 mm above the table. Inside it: two stepped cradle ribs for the motor, two
low axial stops, a notch in the nose wall for the spindle, and a pocket with a standoff for the trigger switch, whose
trigger leaves through a slot in the handle's front wall. On the left, on a two-rail kitting tray: the motor-gearbox
unit (an RS-550-size can and a gearbox, 36 mm across, 131 mm long with its spindle, 315 g). On the right: the
trigger switch (black, 30 x 24 x 22 mm, grey trigger). In front of the nest, open side down on the bench: the right
half (the cover), 110 g.

Final assembly of the housing: lay the unit in its ribs with the spindle through the nose, set the switch in its
pocket with the trigger through its slot, then close the right half down onto the left half with both hands so it
meets it all round. The scorer reads the end state (scoring.py).

The scorer reads the arms (TCP, pad tips, gripper opening) through rigstate.py, which wraps the embodiment's step and
reset from this file's import; it changes nothing the embodiment does. The three loose parts are compound bodies
from mechkit.py (a copy of tools/kit/mechkit.py, version 1 features only: compound bodies, cylinders lying along x).

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

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena)
import offline_model  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Close up the drill housing. Lay the motor and gearbox in the open orange half, spindle through the nose, "
    "and set the trigger switch in its pocket with the trigger through its slot. Then close the other half down "
    "onto it so the two halves meet all round."
)


def _centre_size(b):
    return tuple((a + c) / 2 for a, c in b), tuple(c - a for a, c in b)


def _fixed(name, b, color):
    (cx, cy, cz), size = _centre_size(b)
    return Box(name, pos=L.world((cx, cy, cz)), size=size, color=color)


LOWER = [_fixed(f"left_{n}", b, L.HOUSING_ORANGE) for n, b in L.LOWER.items()]
NEST = [_fixed(n, b, L.NEST_GREY) for n, b in L.NEST.items()]
TRAY = [Box(f"tray_rail_{i}", pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.TRAY_H / 2),
            size=(x1 - x0, y1 - y0, L.TRAY_H), color=L.TRAY_GREY)
        for i, ((x0, x1), (y0, y1)) in enumerate(L.TRAY_RAILS)]

_UNIT_COLOR = {"spindle": L.SPINDLE_STEEL, "gearbox": L.GEARBOX_BLACK, "can": L.CAN_SILVER, "boss": L.CAN_SILVER}
MOTOR = mk.assembly(
    "motor", origin=L.UNIT_START, mass=L.UNIT_MASS, color=L.CAN_SILVER,
    shapes=[mk.cylinder(((x0 + x1) / 2, 0.0, 0.0), r, x1 - x0, axis="X", color=_UNIT_COLOR[n])
            for n, x0, x1, r in L.UNIT_PARTS])
SWITCH = mk.assembly(
    "switch", origin=L.SWITCH_START, mass=L.SWITCH_MASS, color=L.SWITCH_BLACK,
    shapes=[mk.box((0.0, 0.0, 0.0), L.SWITCH_BODY), mk.box(L.TRIGGER[0], L.TRIGGER[1], color=L.TRIGGER_GREY)])
COVER = mk.assembly(
    "cover", origin=L.COVER_START, mass=L.COVER_MASS, color=L.HOUSING_ORANGE,
    shapes=[mk.box(*_centre_size(b)) for b in L.COVER.values()])
OBJECTS = [*LOWER, *NEST, *TRAY, MOTOR, SWITCH, COVER]

_TRACKER = Tracker()
offline_model.register(MOTOR, SWITCH, COVER)
_OFFLINE = offline_model.OfflineParts()  # offline checks only (no simulator): the parts' own shapes, not their boxes


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    _OFFLINE.reset()


def _no_simulator(objects) -> bool:
    """True only under tools/offline/check_task.py, whose states are built from bounding boxes."""
    return mk.env() is None and getattr(objects.get("motor"), "source", "sim") == "offline"


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _turns(name: str, quat_xyzw) -> dict:
    """Trace only: the part's x axis turned about the vertical (yaw, + toward +y) and tipped (pitch, + nose up), deg."""
    try:
        x, y, z, w = (float(v) for v in quat_xyzw)
        ax = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w))
        return {f"{name}_yaw_deg": round(math.degrees(math.atan2(ax[1], ax[0])), 2),
                f"{name}_pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, ax[2])))), 2)}
    except (TypeError, ValueError):
        return {}


def _loose(objects) -> dict:
    return {n: objects[n] for n in L.LOOSE}


def success(objects) -> bool:
    """Unit and switch seated, the halves closed all round, nothing held and both arms readable; held 1 s. This runs
    first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    offline = _no_simulator(objects)
    if offline:  # no simulator: the parts move by the kinematic model in offline_model.py
        _OFFLINE.step(arms)
        objects = _OFFLINE.states(objects)
    loose = _loose(objects)
    result = _TRACKER.update(loose, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in loose.items():
        row[f"{n}_origin"] = getattr(s, "origin", s.pos)
        row[f"{n}_quat"] = s.quat_xyzw
        row.update(_turns(n, s.quat_xyzw))
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    if offline:
        row["offline_events"] = [list(map(str, e)) for e in _OFFLINE.events[-4:]]
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the right half lies on the housing, let go, while a part inside it is not seated. Else 1.0 unit and
    switch seated and the halves closed; 0.7 both seated and the right half lying on the housing but not closed;
    otherwise 0.3 for the seated unit plus 0.2 for the seated switch. A step whose arms cannot be read scores 0."""
    if _no_simulator(objects):
        objects = _OFFLINE.states(objects)
    return _TRACKER.result(_loose(objects), rigstate.arms()).score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap. Grip 8 mm narrower than the part; let a part go by
# opening 12 mm wider than it (6 mm per side); let the cover go fully open. Yaw 0 closes the fingers along world y.
OPEN = -0.04695
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
TIPS_ABOVE_AXIS = 0.0015  # the unit and the switch are held with the pad tips 1.5 mm above the parting plane's level
PLACE_GAP = 0.0015  # a part is let go 1.5 mm above its seat
RELEASE_EXTRA = 0.012


def _grip(w: float) -> float:
    return -(w - 0.008) / 2


def _release(w: float) -> float:
    return -(w + RELEASE_EXTRA) / 2


UNIT_W = 2 * math.sqrt(L.R_UNIT**2 - TIPS_ABOVE_AXIS**2)  # the can's chord at the pad tips' level (35.9 mm)
UNIT_GRIP, UNIT_RELEASE = _grip(UNIT_W), _release(UNIT_W)
SWITCH_GRIP, SWITCH_RELEASE = _grip(L.SWITCH_BODY[1]), _release(L.SWITCH_BODY[1])  # across its 24 mm (y)
COVER_GRIP_L = _grip(L.BARREL_Y[1] - L.BARREL_Y[0])  # left: across the barrel's 56 mm (y)
COVER_GRIP_R = _grip(L.FOOT_Y[1] - L.FOOT_Y[0])  # right: across the battery foot's 30 mm (y)

UNIT_COM_X = float(MOTOR.center_of_mass()[0])  # the unit is held over its centre of mass (no torque about the pinch)
TCP_ABOVE_AXIS = PAD_TIPS + TIPS_ABOVE_AXIS  # 14.5 mm
UNIT_PICK = (L.UNIT_START[0] + UNIT_COM_X, L.UNIT_START[1], L.UNIT_START[2] + TCP_ABOVE_AXIS)
# Demo trims from the GPU trace (job 20261119T002221), added to the commanded place points only. Carrying the 315 g
# unit, the left arm sat 2.5 mm short in x and 1.3 mm left in y at the 2 cm settle (TCP sagging 7.3 mm), up to
# (-3.4, +2.1) mm while lowering, and the grip turned the unit +1.5 deg about the vertical; it landed 3.3 mm toward
# the rear stop (3.5 mm clear) with its spindle on the nose wall (1 mm clear), 6.5 mm high. The cover, carried by
# both arms, turned with the pair of grips +1.0 deg (left TCP (-1.8, +1.0) mm, right (+0.9, -1.2) mm off at the
# 12 mm settle), which leaves its 3 mm tongue (2 mm play) about 0.1 mm off the rear wall. Units: m and rad.
UNIT_TRIM = (0.0027, -0.0015)
UNIT_PLACE_YAW = math.radians(-1.5)
COVER_TRIM = {"left": (0.0018, -0.0010), "right": (-0.0009, 0.0012)}
UNIT_PLACE = L.world((UNIT_COM_X + UNIT_TRIM[0], UNIT_TRIM[1], TCP_ABOVE_AXIS + PLACE_GAP))
SWITCH_PICK = (L.SWITCH_START[0], L.SWITCH_START[1], L.SWITCH_START[2] + TCP_ABOVE_AXIS)
SWITCH_PLACE = L.world((L.SWITCH_SEAT[0], L.SWITCH_SEAT[1], TCP_ABOVE_AXIS + PLACE_GAP))

# The cover is held by both arms through its outer walls, pad tips 3 mm above its rim: the left across the barrel
# 50 mm toward the nose from O, the right across the battery foot's rear end, behind the handle (19 cm apart: each
# YAM wrist housing stands about 7 cm out beyond its TCP, so grips 13.6 cm apart, barrel and handle end, put the two
# wrists 26 mm into each other in the capsule proxy); the cover's centre of mass is 9 mm off the line between them.
COVER_TIPS = 0.003
COVER_GRIP_AT = {"left": (0.050, 0.0), "right": (-0.078, (L.FOOT_Y[0] + L.FOOT_Y[1]) / 2)}  # cover frame (x, y)
COVER_SHIFT = L.OX - L.COVER_START[0]  # the cover travels this far along +x, nothing else
COVER_LIFT_RIM = L.TOP + 0.060  # carried with its rim 60 mm up: its tongue 14 mm over the unit's top
COVER_NEAR_RIM = L.TOP + L.PARTING + 0.012
COVER_DOWN_RIM = L.TOP + L.PARTING + 0.0005
CARRY_Z = L.TOP + 0.10
YAW = {"left": 0.0, "right": 0.0}  # cover grips: both close along y
PARK_L = (0.220, 0.200)  # the left arm waits here (10 cm up) while the right sets the switch


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _pick(p, grip, what, yaw=0.0) -> list[Waypoint]:
    x, y, z = p
    return [
        Waypoint(2.0, (x, y, CARRY_Z), OPEN, f"over the {what}", yaw),
        Waypoint(2.0, (x, y, z), OPEN, f"down round the {what}", yaw),
        Waypoint(0.5, (x, y, z), OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, z), grip, f"grip the {what}", yaw),
        Waypoint(1.5, (x, y, CARRY_Z), grip, "lift", yaw),
    ]


def _place(p, grip, release, what, yaw=0.0, settle=0.5, lower=2.0) -> list[Waypoint]:
    x, y, z = p
    return [
        Waypoint(2.0, (x, y, CARRY_Z), grip, f"carry the {what} over its seat", yaw),
        Waypoint(1.2, (x, y, z + 0.020), grip, "down to 2 cm above", yaw),
        Waypoint(settle, (x, y, z + 0.020), grip, "settle", yaw),
        Waypoint(lower, (x, y, z), grip, "lower the last 2 cm slowly", yaw),
        Waypoint(0.5, (x, y, z), grip, "settle", yaw),
        Waypoint(1.5, (x, y, z), release, f"let go of the {what}", yaw),
        Waypoint(1.0, (x, y, CARRY_Z), release, "straight up", yaw),
    ]


def _wait(until: float, wps: list[Waypoint], label: str) -> list[Waypoint]:
    """Hold the last pose until the plan's clock reaches ``until`` seconds."""
    w = wps[-1]
    return [Waypoint(max(until - _span(wps), 0.0), w.pos, w.finger, label, w.yaw)]


def _cover_point(arm: str, rim_z: float, shift: float, trim: bool = False) -> tuple[float, float, float]:
    gx, gy = COVER_GRIP_AT[arm]
    tx, ty = COVER_TRIM[arm] if trim else (0.0, 0.0)
    return (L.COVER_START[0] + shift + gx + tx, L.COVER_START[1] + gy + ty, rim_z + COVER_TIPS + PAD_TIPS)


def _cover(arm: str) -> list[Waypoint]:
    """The cover, both arms on one clock: down round its walls, grip, lift it level, carry it 15.5 cm along x over the
    housing (to the trimmed points), lower it (the last 12 mm slowly) until its rim meets the left half's, settle,
    let go fully open."""
    grip, yaw = (COVER_GRIP_L if arm == "left" else COVER_GRIP_R), YAW[arm]
    start_rim = L.COVER_START[2]
    above = (*_cover_point(arm, start_rim, 0.0)[:2], CARRY_Z)
    at = _cover_point(arm, start_rim, 0.0)
    lifted = _cover_point(arm, COVER_LIFT_RIM, 0.0)
    over = _cover_point(arm, COVER_LIFT_RIM, COVER_SHIFT, trim=True)  # trimmed over the housing (GPU trace)
    near = _cover_point(arm, COVER_NEAR_RIM, COVER_SHIFT, trim=True)
    down = _cover_point(arm, COVER_DOWN_RIM, COVER_SHIFT, trim=True)
    head = [Waypoint(2.0, above, OPEN, "over the cover's grip", yaw)]
    if arm == "right":  # the right arm gets there first and waits, so the two arms never move past each other
        head.append(Waypoint(COVER_STAGGER, above, OPEN, "wait for the left arm", yaw))
    return head + [
        Waypoint(2.0, at, OPEN, "down outside its walls", yaw),
        Waypoint(0.5, at, OPEN, "settle", yaw),
        Waypoint(1.5, at, grip, "grip the cover", yaw),
        Waypoint(0.5, at, grip, "settle", yaw),
        Waypoint(2.0, lifted, grip, "lift it level", yaw),
        Waypoint(3.0, over, grip, "carry it over the housing", yaw),
        Waypoint(0.5, over, grip, "settle", yaw),
        Waypoint(1.5, near, grip, "down to 12 mm above the rim", yaw),
        Waypoint(0.5, near, grip, "settle", yaw),
        Waypoint(2.5, down, grip, "lower the last 12 mm slowly", yaw),
        Waypoint(0.5, down, grip, "settle", yaw),
        Waypoint(1.5, down, OPEN, "let go", yaw),
        Waypoint(1.0, (*down[:2], CARRY_Z), OPEN, "straight up", yaw),
    ]


def _left_motor() -> list[Waypoint]:
    wps = _pick(UNIT_PICK, UNIT_GRIP, "motor")
    wps += _place(UNIT_PLACE, UNIT_GRIP, UNIT_RELEASE, "motor", UNIT_PLACE_YAW, settle=1.0, lower=3.0)
    wps.append(Waypoint(1.5, (*PARK_L, CARRY_Z), OPEN, "out of the way", 0.0))
    return wps


LEFT_OUT_S = _span(_left_motor())


def _right_switch() -> list[Waypoint]:
    wps = _pick(SWITCH_PICK, SWITCH_GRIP, "switch")
    wps += _wait(LEFT_OUT_S, wps, "wait until the left arm is out of the way")
    wps += _place(SWITCH_PLACE, SWITCH_GRIP, SWITCH_RELEASE, "switch")
    return wps


COVER_T0 = _span(_right_switch())  # the right arm starts the cover here, the left COVER_STAGGER later
COVER_STAGGER = 2.0


def _left() -> list[Waypoint]:
    """Left arm: the motor-gearbox unit into its ribs, wait out of the way, then its side of the cover."""
    wps = _left_motor()
    wps += _wait(COVER_T0 + COVER_STAGGER, wps, "wait for the right arm")
    return wps + _cover("left")


def _right() -> list[Waypoint]:
    """Right arm: the switch into its pocket (after the left arm has left the housing), then its side of the cover."""
    return _right_switch() + _cover("right")


def _gpu_arm_error(side: str, part: str, tips) -> tuple[float, float]:
    """Offline only (offline_model.py): within 20 mm of a trimmed place point, the GPU arm carries the part off its
    command by minus the demo's trim there (what the trace of job 20261119T002221 measured)."""
    zones = [("left", "motor", UNIT_PLACE, UNIT_TRIM)]
    zones += [(a, "cover", _cover_point(a, COVER_DOWN_RIM, COVER_SHIFT, trim=True), COVER_TRIM[a]) for a in YAW]
    for s, n, p, (tx, ty) in zones:
        if s == side and n == part and math.hypot(tips[0] - p[0], tips[1] - p[1]) < 0.020:
            return -tx, -ty
    return 0.0, 0.0


_OFFLINE.arm_error = _gpu_arm_error


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
