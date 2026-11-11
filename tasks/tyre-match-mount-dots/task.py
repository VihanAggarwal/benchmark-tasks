"""On the station table: a tyre fitter's bench at about 1:10. Two silver steel wheels (36 mm across, 24 mm tall), each
standing on its own grey changer turntable (54 mm across, 6 mm tall), 34 cm in front of the arm bases and 15 cm either
side of the midline. Each wheel has a black valve stem (a 6 mm post, 8 mm tall) on its top face, 10 mm off its axis
toward the far side of the table; wheel A (left) also has a white low-point mark (an 8 mm painted dot) on its top
face, 10 mm off its axis at 120 degrees. In front of each wheel lies a loose black rubber tyre (64 mm across, a 46 mm
bore, 22 mm tall) with a red dot and a yellow dot, 100 degrees apart, on its top sidewall. Rule (tyre makers'
match-mounting): on a wheel with a low-point mark the red dot goes to the mark; on a wheel without one the yellow dot
goes to the valve. The robot mounts a tyre on each wheel (down on the turntable round the wheel), turned so, and
lets go.

The tyres are mechkit assemblies (mechkit.py, next to this file): each one loose rigid body of 36 overlapping upright
rubber ribs in a ring and two thin painted dots, so its bore is a real hole that drops over the wheel and the scorer
can read its pose (base frame and orientation). The scorer also reads the arms (TCP, pad-tip point, opening) through
rigstate.py, which wraps the embodiment's step and reset from this file's import; neither changes anything else the
embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: each arm mounts the tyre in front of it on the wheel on its own side; DEMO_IDLE_ARM=left (or right) leaves
that arm at its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so each tyre is built as one compound body)
import rigstate  # noqa: E402
from scoring import Tracker, dot_angle_deg, heading_deg, valid  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Cylinder  # noqa: E402

_RIG_HOOKED = rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it)
if not _RIG_HOOKED and not getattr(sys.modules.get("yam_arena"), "OFFLINE_STUB", False):
    print("[tyre-match-mount-dots] WARNING: rigstate could not hook yam_arena's embodiment, so the grippers cannot "
          "be read: no tyre can count as let go", file=sys.stderr)

SCENE = "station"
SCORE_OVER = "end"
# Both tyres seated (let go, still for 1 s) with the right dots at the right marks, held 1 s: the trial cannot end
# while a gripper still holds or touches a tyre.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Mount a tyre on each wheel on the changer turntables, turning each tyre so its marks line up with the wheel the "
    "way tyre makers require, then let go. The red and yellow dots are on the tyre sidewalls. The small post on each "
    "wheel is the valve stem, and the white dot on one wheel is its low-point mark."
)


# -- the objects --------------------------------------------------------------------------------------------------
def _fixed_cylinder(name, x, y, z0, radius, height, color) -> Cylinder:
    """A fixed upright cylinder standing on height z0 above the table top."""
    return Cylinder(name, pos=(x, y, L.TOP + z0 + height / 2), radius=radius, height=height, color=color)


TURNTABLES, WHEELS, VALVES, MARKS = [], [], [], []
for _w, (_x, _y, _marked) in L.WHEELS.items():
    _side = _w[-1]
    TURNTABLES.append(_fixed_cylinder(f"turntable_{_side}", _x, _y, 0.0, L.TURNTABLE_RADIUS, L.TURNTABLE_H,
                                      L.TURNTABLE_GREY))
    WHEELS.append(_fixed_cylinder(_w, _x, _y, L.TT_TOP, L.WHEEL_RADIUS, L.WHEEL_H, L.WHEEL_SILVER))
    _vx, _vy = L.on_top(L.VALVE_R, L.VALVE_ANGLE)
    VALVES.append(_fixed_cylinder(f"valve_{_side}", _x + _vx, _y + _vy, L.WHEEL_TOP, L.VALVE_RADIUS, L.VALVE_H,
                                  L.VALVE_BLACK))
    if _marked:
        _mx, _my = L.on_top(L.MARK_R, L.MARK_ANGLE)
        MARKS.append(_fixed_cylinder(f"low_point_mark_{_side}", _x + _mx, _y + _my, L.WHEEL_TOP, L.MARK_RADIUS,
                                     L.MARK_H, L.MARK_WHITE))


def tyre_shapes() -> list:
    """The 36 ribs of the ring and the two painted dots on its top sidewall, in the tyre's base frame (the centre of
    its bottom face; local +x toward the red dot)."""
    shapes = []
    for k in range(L.N_RIBS):
        dx, dy = L.on_top(L.RIB_R, L.RIB_OFFSET_DEG + k * L.RIB_STEP_DEG)
        shapes.append(mk.cylinder((dx, dy, L.TYRE_H / 2), L.RIB_RADIUS, L.TYRE_H))
    for angle, colour in ((L.RED_LOCAL_DEG, L.DOT_RED), (L.YELLOW_LOCAL_DEG, L.DOT_YELLOW)):
        dx, dy = L.on_top(L.DOT_R, angle)
        shapes.append(mk.cylinder((dx, dy, L.TYRE_H + L.DOT_H / 2), L.DOT_RADIUS, L.DOT_H, color=colour))
    return shapes


TYRES = [
    mk.assembly(name, origin=(x, y, L.TOP), yaw=math.radians(yaw), mass=L.TYRE_MASS, color=L.TYRE_BLACK,
                shapes=tyre_shapes())
    for name, (x, y, yaw) in L.TYRES.items()
]
OBJECTS = [*TURNTABLES, *WHEELS, *VALVES, *MARKS, *TYRES]

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


def _tyres(objects) -> dict:
    get = objects.get if hasattr(objects, "get") else (lambda k: None)
    return {n: get(n) for n in L.TYRE_NAMES}


def success(objects) -> bool:
    """A tyre seated on each wheel (down on its turntable round the wheel, flat, dots up, let go: both grippers'
    finger zones 10 mm or more from it, still for 1 s) with, on wheel A (marked), the red dot within 10 degrees of the
    white mark and, on wheel B (no mark), the yellow dot within 10 degrees of the valve; held 1 s. This runs first
    every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    tyres = _tyres(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(tyres, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    for n, t in tyres.items():
        if valid(t):
            row[f"{n}_origin"] = t.origin
            row[f"{n}_quat"] = t.quat_xyzw
            row[f"{n}_source"] = getattr(t, "source", None)
            row[f"{n}_yaw_deg"] = round(heading_deg(t), 2)
            row[f"{n}_red_deg"] = round(dot_angle_deg(t, "red"), 2)
            row[f"{n}_yellow_deg"] = round(dot_angle_deg(t, "yellow"), 2)
        else:
            row[f"{n}_state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips, openings=[arms["left"].opening, arms["right"].opening])
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """Per wheel: 'correct' when exactly one tyre is seated on it (down on its turntable round the wheel, underside
    within 3 mm of the turntable's top, axis within 5 degrees of vertical, dots up, let go and still for 1 s) with the
    rule's dot within 10 degrees of the rule's mark (wheel A: red dot to the white mark; wheel B: yellow dot to the
    valve). 1.0 both wheels correct; 0.5 one; 0.25 at least one tyre seated but no wheel correct; 0 no tyre seated.
    Read from the state at hand (the last step)."""
    return _TRACKER.result(_tyres(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Each arm mounts the tyre in front of it on the wheel on its own side, both at once (the wheels are 30 cm apart):
# the left arm tyre_a on wheel A (red dot from 60 to 120 degrees: a 60-degree turn to the left), the right arm tyre_b
# on wheel B (yellow dot from 60 to 0 degrees: a 60-degree turn to the right).
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.038  # 76 mm: 6 mm either side of the 64 mm tyre
GRIP = -(L.TYRE_OD - 0.008) / 2  # 56 mm commanded across the tyre's 64 mm
# Grip yaw: Waypoint yaw 0 closes the fingers along world y, so the closing axis points at yaw + 90 degrees. Each grip
# closes across the tyre's local 0/180 or 90/270 line (between two pairs of ribs: a stable grip), with the yaw picked
# so the arm's wrist (joint6) stays well inside its limits at the pick and after the turn.
TURN = {"tyre_a": +60.0, "tyre_b": -60.0}  # degrees to turn each tyre: to L.required_yaw(wheel) from its start
PICK_YAW_DEG = {"tyre_a": -30.0, "tyre_b": 50.0}  # closing axes at 60 and 140 degrees: each tyre's local 0/180 line
# Trim of the place yaw, for the GPU run: parts carried in a gripper have ended turned counter-clockwise of the
# commanded yaw by +0.6 to +1.7 degrees in our GPU runs, so each place yaw is set 1 degree clockwise (offline, with no
# such turn, the dots land 1 degree clockwise of their marks; the window is 10 degrees either way). If the trace's
# rule_error_deg at rest is more than 3 degrees, move YAW_TRIM_DEG by minus the signed error.
YAW_TRIM_DEG = -1.0
GRIP_UP = 0.016  # the TCP 16 mm above the tyre's underside: pad tips 3 mm above it (on the table at the pick)
HOVER = L.TOP + 0.100  # carried with the tyre's underside 84 mm up: 46 mm over the valve stems' tops
ENTRY_GAP = 0.010  # the slow descent starts with the tyre's underside this far over the valve stem's top
SEAT_GAP = 0.001  # let go with the tyre's underside this far over the turntable's top
# Trim of the place point (x, y), for the GPU run: arms under load have trailed their targets and set parts down a
# few mm short along the carry (about 4 mm; here +x). The tyre's bore clears the wheel by 5 mm all round, so the place
# point leads the axis by 2.5 mm: any trail from -2.5 to +7.5 mm still drops the bore over the wheel (and offline, with
# no trail, the bore clears the wheel by 2.5 mm). If the trace shows the bore catching on the wheel's top edge
# (bottom_mm stuck near 24 with tilt_deg over 5), move PLACE_TRIM toward the side it caught on.
PLACE_TRIM = (0.0025, 0.0)
UP = L.TOP + 0.120
PARK = {"left": (0.200, 0.240, L.TOP + 0.150), "right": (0.200, -0.240, L.TOP + 0.150)}
ARM_OF = {"tyre_a": "left", "tyre_b": "right"}
WHEEL_OF = {"tyre_a": L.MARKED_WHEEL, "tyre_b": L.PLAIN_WHEEL}


def place_yaw_deg(name: str) -> float:
    return PICK_YAW_DEG[name] + TURN[name] + YAW_TRIM_DEG


def tyre_plan(name: str, wheel: str | None = None, turn: float | None = None, release: bool = True,
              trim=PLACE_TRIM) -> list[Waypoint]:
    """Pick ``name`` at its start across its outside, lift it, carry it over ``wheel`` (default: the one on its
    arm's side) turning it ``turn`` degrees (default: the rule's turn), lower it over the wheel onto the turntable
    and, with ``release``, let go and go up and aside (the tests use other wheels, turns and no release)."""
    sx, sy, _ = L.TYRES[name]
    wheel = WHEEL_OF[name] if wheel is None else wheel
    turn = TURN[name] if turn is None else turn
    wx, wy = L.wheel_axis(wheel)
    px, py = wx + trim[0], wy + trim[1]
    y0 = math.radians(PICK_YAW_DEG[name])
    y1 = math.radians(PICK_YAW_DEG[name] + turn + YAW_TRIM_DEG)
    pick_z = L.TOP + GRIP_UP
    entry_z = L.TOP + L.VALVE_TOP + ENTRY_GAP + GRIP_UP
    seat_z = L.TOP + L.TT_TOP + SEAT_GAP + GRIP_UP
    plan = [
        Waypoint(2.0, (sx, sy, HOVER), OPEN, f"over {name}, open", y0),
        Waypoint(1.0, (sx, sy, L.TOP + 0.050), OPEN, "down to 5 cm", y0),
        Waypoint(1.5, (sx, sy, pick_z), OPEN, "slowly down beside it", y0),
        Waypoint(0.5, (sx, sy, pick_z), OPEN, "settle", y0),
        Waypoint(1.5, (sx, sy, pick_z), GRIP, "grip it across its outside", y0),
        Waypoint(1.5, (sx, sy, HOVER), GRIP, "lift it", y0),
        Waypoint(2.5, (px, py, HOVER), GRIP, f"carry it over {wheel}, turning it", y1),
        Waypoint(0.8, (px, py, HOVER), GRIP, "settle over the wheel", y1),
        Waypoint(1.0, (px, py, entry_z), GRIP, "down: its underside 1 cm over the valve stem", y1),
        Waypoint(0.5, (px, py, entry_z), GRIP, "settle", y1),
        Waypoint(2.5, (px, py, seat_z), GRIP, "slowly down over the wheel to 1 mm over the turntable", y1),
        Waypoint(0.8, (px, py, seat_z), GRIP, "settle", y1),
    ]
    if release:
        plan += [
            Waypoint(1.5, (px, py, seat_z), OPEN, "let go", y1),
            Waypoint(1.5, (px, py, UP), OPEN, "straight up", y1),
            Waypoint(1.5, PARK[ARM_OF[name]], OPEN, "aside", y1),
        ]
    else:
        plan.append(Waypoint(6.0, (px, py, seat_z), GRIP, "keep holding it", y1))
    return plan


def demo():
    """Both arms at once: the left arm mounts tyre_a on wheel A, the right arm tyre_b on wheel B. DEMO_HOLD_ONLY=1: no
    plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the
    one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": tyre_plan("tyre_a"), "right": tyre_plan("tyre_b")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
