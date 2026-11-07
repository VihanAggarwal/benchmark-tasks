"""On the station table: the support plate of an SMT pick-and-place machine (fixed, grey steel, 124 x 220 x 5 mm)
centred 30 cm in front of the arm bases on the midline, between two low blue conveyor rails running along y; on it
a white outline of the board that will stop there (120 x 160 mm), split into quarters by two white lines, and red
areas where the board has parts on its underside, leaving one clear window per quarter, each in a different place;
and six loose steel support pins (20 mm across, 45 mm tall, 110 g), three standing in a black tray on each side.
Task: one pin upright in each quarter, at least 5 mm inside the outline and 3 mm clear of every red area, let go
with both grippers moved away; no pin touching a red area or at a rail.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s: one supporting pin in each quarter, each let go (no finger zone reaching it) and still.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Set up the board supports for the next job: stand one support pin upright in each quarter of the white board "
    "outline (the thin white lines split it into quarters), the whole pin at least 5 mm inside the outline and at "
    "least 3 mm clear of every red area, where the board has parts on its underside that a pin would damage. Keep "
    "every pin at least 5 mm from the blue conveyor rails. Take the pins from the black trays, let go of them and "
    "move both grippers away from the pins."
)


def _flat(name: str, rect, z0: float, h: float, color) -> Box:
    x0, x1, y0, y1 = rect
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, z0 + h / 2), size=(x1 - x0, y1 - y0, h), color=color)


PLATE = Box("support_plate", pos=(L.PLATE_X, L.PLATE_Y, L.TOP + L.PLATE_SIZE[2] / 2), size=L.PLATE_SIZE,
            color=L.PLATE_GREY)
RAILS = [Box(name, pos=(x, L.PLATE_Y, L.TOP + L.RAIL_H / 2), size=(L.RAIL_W, L.RAIL_LEN, L.RAIL_H),
             color=L.RAIL_BLUE) for name, x in L.RAILS]
LINES = [_flat(name, (x0, x1, y0, y1), L.PLATE_TOP, L.MARK_H, L.LINE_WHITE) for name, x0, x1, y0, y1 in L.lines()]
RED = [_flat(name, (x0, x1, y0, y1), L.PLATE_TOP, L.MARK_H, L.RED_AREA) for name, x0, x1, y0, y1 in L.RED]
TRAYS = [Box(f"pin_tray_{side}", pos=(L.TRAY_X, sign * L.TRAY_Y, L.TOP + L.TRAY_SIZE[2] / 2), size=L.TRAY_SIZE,
             color=L.TRAY_BLACK) for side, sign in (("left", 1.0), ("right", -1.0))]
PINS = [Cylinder(name, pos=(x, y, L.TRAY_TOP + L.PIN_H / 2), radius=L.PIN_R, height=L.PIN_H, mass=L.PIN_MASS,
                 color=L.PIN_STEEL) for name, x, y in L.PINS]
OBJECTS = [PLATE, *RAILS, *LINES, *RED, *TRAYS, *PINS]

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


def _pins(objects) -> dict:
    return {n: objects[n] for n in L.PIN_NAMES if n in objects}


def success(objects) -> bool:
    """One supporting pin in each quarter (upright on the plate, 5 mm inside the outline, 3 mm clear of red, let
    go), no pin touching a red area or at a rail; the supporting pins still for 1 s. This runs first every control
    step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    states = _pins(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(states, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in states.items():
        row[f"{n}_pos"] = getattr(s, "pos", None)
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any pin touches a red area (outline over it, lowest point within 5 mm of the plate) or is at a rail
    (outline within 5 mm of it, not more than 5 mm above its top). Else 0.25 for each quarter with a pin standing
    upright on the plate in it, its whole outline at least 5 mm inside the outline and 3 mm clear of red, let go."""
    return _TRACKER.result(_pins(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PIN_GRIP = -(2 * L.PIN_R - 0.008) / 2  # 12 mm commanded across the 20 mm pin
LET_GO = -0.020  # 40 mm apart: 10 mm each side of the pin, so the open pads stay well away from its neighbours
YAW = 0.0  # every grip closes the fingers along y: along the tray rows' neighbours never, and never toward a rail
GRIP_UP = 0.032  # the TCP this far above a pin's bottom: pad tips 19 mm up, the pads on its upper 26 mm
CARRY_Z = L.TOP + 0.12  # a carried pin's bottom 88 mm above the table, 38 mm over the standing pins' tops
PARK_Z = L.TOP + 0.15
ENTRY = 0.020  # the last 20 mm down are taken slowly
PICK_Z = L.TRAY_TOP + GRIP_UP
PLACE_Z = L.PLATE_TOP + 0.001 + GRIP_UP  # let go with the pin's bottom 1 mm above the plate
# Where each pin goes: the middle of the quarter's clear window (where a pin's centre may stand and still be 5 mm
# inside the outline and 3 mm clear of red: +-7 to +-8.3 mm each way round these points). The arms are joint
# springs and the controller does not carry a payload's weight; scaled from the sag measured in simulation for a
# 200 g part held the same way ((-1.9, +-1.9, -5.6) mm), a 110 g pin sags about (-1, +-1, -3) mm, well inside the
# window, so no trim is needed.
# The pin touches the plate about 2 mm before the slow descent ends. Read each pin's red_mm and inside_mm in
# TASK_TRACE.
TARGETS = {
    "near_left": (0.2630, 0.0570),
    "near_right": (0.2700, -0.0400),
    "far_left": (0.3168, 0.0178),
    "far_right": (0.3300, -0.0570),
}
LEFT_PARK = (0.20, 0.25, PARK_Z)
RIGHT_PARK = (0.20, -0.25, PARK_Z)


def _pick(name: str, approach: float) -> list[Waypoint]:
    """Over a pin in its tray, slowly down round it, grip it near its top, lift it straight up."""
    x, y = L.STARTS[name]
    return [
        Waypoint(approach, (x, y, CARRY_Z), OPEN, f"over {name}", YAW),
        Waypoint(1.5, (x, y, PICK_Z + ENTRY), OPEN, "down to 2 cm above", YAW),
        Waypoint(1.0, (x, y, PICK_Z), OPEN, "slowly down round the pin", YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, PICK_Z), PIN_GRIP, "grip the pin", YAW),
        Waypoint(1.2, (x, y, CARRY_Z), PIN_GRIP, "lift it straight up", YAW),
    ]


def _place(quarter: str) -> list[Waypoint]:
    """Carry the pin high over the quarter's clear window, stop, lower the last 2 cm slowly, let go, go up."""
    x, y = TARGETS[quarter]
    return [
        Waypoint(1.8, (x, y, CARRY_Z), PIN_GRIP, f"carry over the {quarter} window", YAW),
        Waypoint(1.2, (x, y, PLACE_Z + ENTRY), PIN_GRIP, "down to 2 cm above the plate", YAW),
        Waypoint(0.8, (x, y, PLACE_Z + ENTRY), PIN_GRIP, "settle", YAW),
        Waypoint(2.0, (x, y, PLACE_Z), PIN_GRIP, "lower the last 2 cm slowly", YAW),
        Waypoint(1.5, (x, y, PLACE_Z), LET_GO, "let go", YAW),
        Waypoint(0.3, (x, y, PLACE_Z), LET_GO, "settle", YAW),
        Waypoint(1.2, (x, y, CARRY_Z), LET_GO, "straight up", YAW),
    ]


def _duration(plan: list[Waypoint]) -> float:
    return sum(w.duration for w in plan)


def _wait_until(plan: list[Waypoint], t: float, label: str) -> list[Waypoint]:
    """Hold the plan's last pose until time t (s from the start), if it gets there earlier."""
    last = plan[-1]
    gap = t - _duration(plan)
    if gap > 1e-9:
        plan = plan + [Waypoint(gap, last.pos, last.finger, label, last.yaw)]
    return plan


def _plans() -> dict[str, list[Waypoint]]:
    """Both arms on one clock, one gripper over the board at a time: the left arm sets near left, the right arm
    near right, the left arm far left, the right arm far right; each arm fetches its next pin from its own tray
    while the other is over the board, and comes over the board only once the other has let go and lifted."""
    left = _pick("pin_left_1", 2.0) + _place("near_left")
    t_left_1 = _duration(left)  # the left arm is up over near left: the board is free
    right = _wait_until(_pick("pin_right_1", 2.0), t_left_1, "hold the pin up by the tray") + _place("near_right")
    t_right_1 = _duration(right)
    left = _wait_until(left + _pick("pin_left_2", 1.8), t_right_1, "hold the pin up by the tray") + _place("far_left")
    t_left_2 = _duration(left)
    right = _wait_until(right + _pick("pin_right_2", 1.8), t_left_2, "hold the pin up by the tray") \
        + _place("far_right")
    # Park with the fingers as they let go (40 mm apart): opening them fully over a pin just let go would widen
    # the finger zone toward it, for nothing.
    left += [Waypoint(2.0, LEFT_PARK, LET_GO, "park to the left", YAW)]
    right += [Waypoint(2.0, RIGHT_PARK, LET_GO, "park to the right", YAW)]
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = _plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
