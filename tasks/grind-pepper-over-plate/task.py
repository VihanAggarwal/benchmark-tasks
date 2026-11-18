"""On the station table, real size: a Peugeot Paris u'Select 12 cm pepper mill (beech, 120 mm tall, 59 mm across)
standing upright on the robot's left, and an IKEA OFTAST 25 cm plate in front of the right arm. The mill is one
mechkit articulation (mechkit.py next to this file): the body (base, loose) and the head on a free spindle about the
mill's axis, with the grinding resistance as joint friction. The plate is a loose cylinder. Nothing is latched or
moved by code: the head turns only when something turns it against the body.

Goal: grind pepper over the middle of the plate, five full turns of the head (clockwise from above, the way the mill
grinds), the mill held clear above the plate; then stand the mill back up on the table, off the plate, and let go.
See scoring.py for the tiers.

The mill's base is loose and its head turns freely on it, so the head can only be turned while the body is held:
one hand holds the body, the other turns the head; a parallel gripper's wrist turns about 225 deg at most, so each
turn of the head takes two strokes, regripping in between.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=on_table | three_turns | leave_on_plate (see demo() below).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built as articulations)
import rigstate  # noqa: E402
import tiltgrip  # noqa: E402
from scoring import HEAD, MILL, PLATE, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 20 steps

INSTRUCTION = ("Grind pepper over the middle of the plate: five full turns of the mill's head. Then stand the mill "
               "back up on the table, off the plate.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _cyls(kind: str, colour):
    return [mk.cylinder((0.0, 0.0, z), r, h, color=colour) for r, z, h in L.mill_shapes(kind)]


def _marks():
    """A dark dot on the crown's rim and a dark line on the shoulder below it, as on a mill marked to count turns
    on video (a dot of paint and a pencil line in the real trial). Both sit within the profile's radius + 0.5 mm."""
    crown_r = L.HEAD_PROFILE[0][1]
    shoulder_r = L.BODY_PROFILE[3][1]
    dot = mk.box(((crown_r - 1.5) * L.MM, 0.0, 0.095), (4 * L.MM, 6 * L.MM, 10 * L.MM), L.MARK)
    line = mk.box(((shoulder_r - 1.0) * L.MM, 0.0, 0.0755), (3 * L.MM, 2 * L.MM, 7 * L.MM), L.MARK)
    return dot, line


_DOT, _LINE = _marks()
HEAD_LINK = mk.spindle(HEAD, "Z", (0.0, 0.0, L.HEAD_PIVOT_Z * L.MM), [*_cyls("head", L.BEECH), _DOT], mass=L.M_HEAD,
                       damping=L.GRIND_DAMPING, friction=L.GRIND_FRICTION, color=L.BEECH)
MILL_SPEC = mk.assembly(MILL, (L.MILL_START[0], L.MILL_START[1], L.TOP), [*_cyls("body", L.BEECH), _LINE],
                        mass=L.M_BODY, joints=[HEAD_LINK], color=L.BEECH, friction=L.MILL_FRICTION)
PLATE_SPEC = Cylinder(PLATE, pos=(L.PLATE_CENTRE[0], L.PLATE_CENTRE[1], L.TOP + L.PLATE_H * L.MM / 2),
                      color=L.PLATE_COLOUR, mass=L.PLATE_MASS, radius=L.PLATE_R * L.MM, height=L.PLATE_H * L.MM)
OBJECTS = [MILL_SPEC, PLATE_SPEC]


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()


def _r(v, n: int = 5) -> list:
    return [round(float(x), n) for x in np.asarray(v, dtype=float).ravel()]


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
    """Five turns ground over the plate, the mill stood back up off the plate, both grippers clear and still for
    the scorer's 20 steps; held 1 s. This runs every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    try:
        m = objects[MILL]
        row["mill"] = {"origin": _r(m.origin), "quat_xyzw": _r(m.quat_xyzw), "head_raw": round(float(m.joints[HEAD]), 5)}
    except Exception as exc:  # noqa: BLE001 - a debugging aid only
        row["mill"] = {"error": str(exc)[:200]}
    if arms is not None:
        try:
            row.update(left_tcp=_r(arms["left"].tcp), right_tcp=_r(arms["right"].tcp),
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.15 per full turn ground over the plate (five at most), plus 0.25 for standing the mill back up off the
    plate and letting go after five; 0 once the mill is dropped or the plate knocked. The latest step's score."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points.
# LEFT holds the body from the robot's side: the stock straight-down grip (pads closing along world y) pitched 70 deg
#   about world y, so it reaches in along +x and a little down; the pads' middle on the mill's axis 49 mm above the
#   base, the mill's centre of mass, so its weight has no lever about the line through the pads (gripped low, it
#   rolls over in the fingers). It lifts the mill, carries it over the plate's centre with its base 15 mm above the
#   plate's top, holds it there through the grinding, then stands it back on its start spot.
# RIGHT turns the head from above, straight down, the pads on the crown's rim, their tips 4 mm below it. Each stroke
#   closes, turns 110 deg clockwise (seen from above), opens and turns back, from yaw +55 to -55 deg: the two fingers
#   then sweep two 110 deg arcs either side of the robot and never the 70 deg gap toward -x where the left gripper
#   is. 20 strokes, 2200 deg: in the simulator the pads slip about 8 deg a stroke, so the head turns about 2020 deg.
OPEN, CLOSED = -0.04695, 0.0  # finger joint targets (m): yam_arena.constants.FINGER_OPEN / FINGER_CLOSED
# Each finger is driven on its own toward its target; told to close right up (0), both saturate at their effort limit
# on the part and nothing centres it between them, so it drifts along the closing axis. The demo closes each finger
# to 3 mm inside the part instead (the pads then squeeze it at about 15 N each, below the limit), so their position
# springs keep the part centred, as the YAM's single rack does on the real gripper.
SQUEEZE = 0.003  # m per finger


def grip_on(width: float) -> float:
    """The finger target (m) that closes the pads to 2 x SQUEEZE inside a part ``width`` m across."""
    return -(width / 2 - SQUEEZE)


BODY_GRIP = grip_on(2 * L.BODY_PROFILE[1][1] * L.MM)  # the lower body, 52 mm
CROWN_GRIP = grip_on(2 * L.HEAD_PROFILE[0][1] * L.MM)  # the crown, 58 mm
LEFT_PITCH = math.radians(70.0)
RIGHT_YAW0 = math.radians(55.0)  # each stroke turns from here down to -55 deg (the wrist reaches -60 to +165 here)
STROKE = math.radians(110.0)
STROKES = 20  # 2200 deg: the pads slip about 8 deg a stroke on the crown, so this ends near 2020 deg
HOLD_Z = 0.045  # m: the mill's base over the table top while it grinds (the plate's top is at 20 mm)
LIFT_Z = 0.065  # m: carrying height of the mill's base
VARIANTS = ("on_table", "three_turns", "leave_on_plate")


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _left_rotation() -> np.ndarray:
    return _ry(-LEFT_PITCH) @ tiltgrip.rotation(0.0, 0.0)


def _left_grip():
    return tiltgrip.TiltedYaw(0.0, _left_rotation())


def _left_approach() -> np.ndarray:
    return _left_rotation()[:, 2]


def _left_tcp(base_xyz) -> np.ndarray:
    """The left TCP holding the body when the mill's base centre is at ``base_xyz`` (world): the pads' middle on the
    axis at the grip height (the TCP is 3 mm short of where the pads' middle meets the axis along the approach)."""
    a = _left_approach()
    c = np.asarray(base_xyz, dtype=float) + np.array([0.0, 0.0, L.body_grip_z()])
    return c - 0.003 * a


def _right_tcp(base_xyz) -> np.ndarray:
    """The right TCP gripping the crown from above when the mill's base centre is at ``base_xyz``."""
    tips = float(base_xyz[2]) + L.HEAD_PROFILE[0][2] * L.MM - 0.004
    return np.array([base_xyz[0], base_xyz[1], tips + 0.013])


def _w(duration, tcp, finger, label, yaw) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in tcp), float(finger), label, yaw)


def left_plan(hold_s: float, variant: str = "") -> list[Waypoint]:
    a = _left_approach()
    g = _left_grip()
    start = np.array([*L.MILL_START, L.TOP])
    up = start + np.array([0.0, 0.0, LIFT_Z])
    over = np.array([*L.PLATE_CENTRE, L.TOP + LIFT_Z])
    hold = np.array([*L.PLATE_CENTRE, L.TOP + HOLD_Z])
    if variant == "on_table":  # the grinding happens with the mill held standing on its start spot
        over, hold = start, start
    grip = _left_tcp(start)
    back = grip - 0.07 * a
    plan = [
        _w(2.0, back + np.array([0.0, 0.0, 0.04]), OPEN, "behind the mill, on the robot's side", g),
        _w(1.0, back, OPEN, "level with the grip, 70 mm out along the approach", g),
        _w(1.2, grip, OPEN, "in: the pads either side of the body", g),
        _w(0.8, grip, BODY_GRIP, "close on the body", g),
        _w(1.2, _left_tcp(up), BODY_GRIP, "lift the mill 65 mm", g),
        _w(2.2, _left_tcp(over), BODY_GRIP, "carry it over the plate's centre", g),
        _w(1.0, _left_tcp(hold), BODY_GRIP, "lower: the base 25 mm over the plate's top", g),
        _w(hold_s, _left_tcp(hold), BODY_GRIP, "hold the body while the head is turned", g),
    ]
    if variant == "leave_on_plate":  # set it down on the plate instead of back on the table
        on_plate = np.array([*L.PLATE_CENTRE, L.TOP + L.PLATE_H * L.MM + 0.002])
        end = on_plate
    else:
        end = start + np.array([0.0, 0.0, 0.002])
    plan += [
        _w(1.0, _left_tcp(over if variant != "on_table" else up), BODY_GRIP, "lift clear of the plate", g),
        _w(2.2, _left_tcp(end + np.array([0.0, 0.0, LIFT_Z])), BODY_GRIP, "carry it back", g),
        _w(1.2, _left_tcp(end), BODY_GRIP, "lower it onto its base", g),
        _w(0.8, _left_tcp(end), OPEN, "let go", g),
        _w(1.0, _left_tcp(end) - 0.05 * a, OPEN, "back out along the approach", g),
        _w(1.5, _left_tcp(end) - 0.05 * a + np.array([0.0, 0.0, 0.18]), OPEN, "straight up, well clear of the mill", g),
        _w(2.0, _left_tcp(end) - 0.05 * a + np.array([0.0, 0.0, 0.18]), OPEN, "hold", g),
    ]
    return plan


LEFT_READY_S = 2.0 + 1.0 + 1.2 + 0.8 + 1.2 + 2.2 + 1.0  # the mill is in place over the plate after this


def right_plan(strokes: int, variant: str = "") -> tuple[list[Waypoint], float]:
    """The right arm's plan, and the time (s from the start) at which it is up off the crown."""
    base = np.array([*L.PLATE_CENTRE, L.TOP + HOLD_Z])
    if variant == "on_table":
        base = np.array([*L.MILL_START, L.TOP])
    grip = _right_tcp(base)
    above = grip + np.array([0.0, 0.0, 0.05])
    plan = [
        _w(LEFT_READY_S - 1.5, np.array([0.20, -0.20, L.TOP + 0.20]), OPEN, "wait clear while the mill comes over",
           RIGHT_YAW0),
        _w(1.5, above, OPEN, "over the crown", RIGHT_YAW0),
        _w(1.0, grip, OPEN, "down: the pads either side of the crown", RIGHT_YAW0),
    ]
    steps = 2  # the 110 deg turn in two 55 deg steps, so each waypoint's slerp turns the short way
    for i in range(strokes):
        plan.append(_w(0.5, grip, CROWN_GRIP, f"stroke {i + 1}: close on the crown", RIGHT_YAW0))
        for k in range(1, steps + 1):
            plan.append(_w(0.7, grip, CROWN_GRIP, f"stroke {i + 1}: turn clockwise", RIGHT_YAW0 - STROKE * k / steps))
        plan.append(_w(0.4, grip, OPEN, f"stroke {i + 1}: open", RIGHT_YAW0 - STROKE))
        for k in range(steps - 1, -1, -1):
            plan.append(_w(0.35, grip, OPEN, f"stroke {i + 1}: turn back, open", RIGHT_YAW0 - STROKE * k / steps))
    plan += [
        _w(0.8, above, OPEN, "up off the crown", RIGHT_YAW0),
        _w(1.2, np.array([0.20, -0.22, L.TOP + 0.20]), OPEN, "clear", RIGHT_YAW0),
    ]
    off_crown = sum(w.duration for w in plan[:-1])  # the time (s) at which the right hand is up off the crown
    return plan, off_crown


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    strokes = 11 if variant == "three_turns" else STROKES  # 1210 deg commanded, about 3.1 turns
    right, off_crown = right_plan(strokes, variant)
    hold_s = off_crown - LEFT_READY_S + 0.3  # the left holds until the right is up off the crown, and a margin
    return {"left": left_plan(hold_s, variant), "right": right}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      on_table        the same grinding with the mill standing on its start spot, not over the plate (0)
      three_turns     only 11 strokes over the plate (about three turns), then stood back (0.45)
      leave_on_plate  five turns, then the mill set down on the plate instead of the table (0.75)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
