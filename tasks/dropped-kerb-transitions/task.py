"""On the station table, a 1:5 model of a kerb line at a new driveway entrance. Two fixed full-height kerbs (120 mm
long, 30 mm tall, battered face darker, toward a dark grey road strip in front) leave a 270 mm gap in the kerb line,
with a red-brown paving strip on the footway behind its middle. In the stock area near the robot lie four loose kerb
units: two low (dropped) kerbs (60 x 25 x 12 mm) and two transition kerbs (60 x 25 mm, stepping 12/21/30 mm along
their length), which are mirror images. Rule: lay the two low kerbs in the gap's middle two slots and a transition in
each end slot rising to the full kerb beside it, every unit bedded flat, its back in line with the full kerbs' backs
and its battered (darker) face to the road; 6 mm joints. Only one transition can rise toward each end with its face
to the road, and each starts on the side of the other end.

The loose units are mechkit compounds (mechkit.py, next to this file): each is one rigid body of two (low kerb) or
six (transition) boxes, so its battered face and high end can be seen and the scorer reads its frame (origin and
orientation). The scorer reads only the four loose units.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the kerb units are built as compound bodies)
from scoring import NAMES, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Held for 2.5 s, not the default 1 s. The scorer cannot see the grippers, so the last unit counts as bedded while
# still gripped 2 mm above the bed, 0.5 s after it stops; with 2.5 s the trial cannot end before the fingers have
# opened (0.3 + 1.5 s after the stop), so the end state is the units lying free.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Lay the dropped kerb for this driveway entrance in the gap in the kerb line, with 6 mm joints: the two low "
    "kerbs in the middle, in front of the red paving, and a transition kerb at each end, rising to the full kerb "
    "beside it so the kerb top has no step at any joint (the two transitions are mirror images). Every kerb goes "
    "battered (darker) face to the road, its back in line with the full kerbs' backs."
)


def _fixed_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


ROAD = _fixed_box("road", L.ROAD_X0, L.ROAD_X1, -L.ROAD_Y, L.ROAD_Y, 0.0, L.ROAD_H, L.ASPHALT)
PAVING = _fixed_box("crossover_paving", L.PAVING_X0, L.PAVING_X1, -L.PAVING_Y, L.PAVING_Y, 0.0, L.PAVING_H,
                    L.PAVING_RED)
FULL_KERBS = []
for _side, _k in L.FULL_KERBS.items():
    _y0, _y1 = sorted((_k * L.GAP_HALF, _k * (L.GAP_HALF + L.FULL_LEN)))
    FULL_KERBS.append(_fixed_box(f"full_kerb_{_side}", L.BACK_X, L.BACK_X + L.BACK_DEPTH, _y0, _y1, 0.0, L.FULL_H,
                                 L.CONCRETE))
    FULL_KERBS.append(_fixed_box(f"full_kerb_{_side}_face", L.BACK_X + L.BACK_DEPTH, L.FACE_X, _y0, _y1, 0.0,
                                 L.FULL_H - L.LIP_DROP, L.FACE_GREY))
UNITS = [
    mk.assembly(
        name,
        origin=(L.UNIT_START[name][0], L.UNIT_START[name][1], L.TOP),
        yaw=L.UNIT_START[name][2],
        mass=L.LOW_MASS if L.KIND[name] == "low" else L.TRANSITION_MASS,
        color=L.CONCRETE,
        shapes=[mk.box(c, s, color=col) for c, s, col in L.unit_shapes(name)],
    )
    for name in NAMES
]
OBJECTS = [ROAD, PAVING, *FULL_KERBS, *UNITS]

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


def success(objects) -> bool:
    """Every slot right: a low kerb bedded in each middle slot and in each end slot a transition rising toward the
    full kerb beside it; bedded = still, flat (tilt <= 5 degrees, lowest point within 3 mm of the bed), both back
    corners within 5 mm of the full kerbs' back line, battered face within 30 degrees of straight at the road, middle
    within 8 mm of the slot's centre; held 2.5 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    states = {n: objects.get(n) for n in NAMES}
    result = _TRACKER.update(states)
    row = {"mechkit_episode": mk.episode(), "step": _TRACKER.steps, "score": result.score,
           "success": result.success, **result.details}
    raw = {}
    for n, s in states.items():
        try:
            raw[n] = {"origin": s.origin, "quat": s.quat_xyzw, "vel": s.lin_vel, "source": getattr(s, "source", None)}
        except AttributeError:
            raw[n] = "unreadable"
    row["raw"] = raw
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each of the four slots holding the right unit bedded the right way: S2 and S3 (the middle) a low
    kerb, S1 and S4 (the ends) a transition rising toward the full kerb beside it. Bedded: still, tilted at most 5
    degrees, lowest point within 3 mm of the bed, both back corners within 5 mm of the line of the full kerbs' backs,
    battered face within 30 degrees of straight at the road, middle within 8 mm of the slot's centre along the line.
    0 if no slot is right. Read from the state at hand (the last step)."""
    return _TRACKER.result({n: objects.get(n) for n in NAMES}).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Each arm lays its own half of the crossing: the left arm S1 (the transition that starts on the right of the
# midline) and S2 (the low kerb on its side), the right arm S3 (its low kerb) and S4 (the transition that starts on
# the left). A unit is gripped at its middle across its 25 mm depth (yaw 0: the fingers close along y, the unit lying
# along x), lifted, turned a quarter turn while it is carried (the turn that brings its battered face round to the
# road) and set down on the kerb line. The arms take turns: only one works at the kerb line at a time (S1, S3, S2,
# S4), and while one reaches across the midline for its transition the other waits parked beside its own base. The
# plan is one clock, with holds.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.0245  # 49 mm: 12 mm each side of the 25 mm depth
GRIP = -(L.UNIT_DEPTH - 0.008) / 2  # 17 mm commanded across the 25 mm depth
PAD = 0.013  # the pad tips are this far below the tool point
GRIP_Z = L.TOP + 0.015  # pad tips 2 mm above the bed: 10 mm of a low kerb's back, 6 mm of its face strip
DROP = 0.002  # units are let go this far above the bed
HOVER = L.TOP + 0.070  # travel height: a carried unit's underside 55 mm above the bed, 25 mm over the tallest kerb
APPROACH = 0.030  # each final descent starts this far above its end point and takes 1.5 s
SIDE = {"left": 1.0, "right": -1.0}
PARK = (0.180, 0.330, L.TOP + 0.100)  # where an arm waits beside its own base while the other reaches across
PARK_YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers along x there: joint 6 22 degrees from its limit
# Each arm's script on the shared clock: ("pick", unit, start), ("place", slot, start), ("park", None, start);
# start = when that step may begin (s). Only one arm is at the kerb line at a time: the left from 8.0 to 17.3 s (S1)
# and 26.6 to 35.9 s (S2), the right from 17.3 to 26.6 s (S3) and from 43.4 s (S4). The left arm reaches across the
# midline first (0-8 s) with the right arm parked; the right arm reaches across last (from 35.9 s) with the left arm
# parked. Both end parked.
SCRIPT = {
    "left": (("pick", "transition_left", 0.0), ("place", "S1", 8.0), ("pick", "low_kerb_1", 15.8),
             ("place", "S2", 26.6), ("park", None, 34.4)),
    "right": (("park", None, 0.0), ("pick", "low_kerb_2", 9.0), ("place", "S3", 17.3), ("park", None, 25.1),
              ("pick", "transition_right", 35.9), ("place", "S4", 43.4), ("park", None, 51.2)),
}
# Trim (x, y) of each unit's place point, for the GPU run: if a unit lands off its aim (in TASK_TRACE back_off_mm
# away from 0 or along_mm away from 0), move its place point by minus the offset.
PLACE_TRIM = {n: (0.0, 0.0) for n in NAMES}


def plan_pairs(side: str) -> list[tuple[str, str]]:
    """(unit, slot) for each unit this arm lays, in order."""
    out, unit = [], None
    for kind, what, _ in SCRIPT[side]:
        if kind == "pick":
            unit = what
        elif kind == "place":
            out.append((unit, what))
    return out


def place_point(unit: str, slot: str) -> tuple[float, float]:
    """Where the unit's middle (and the tool point) goes: on the kerb line's middle, at the slot's centre."""
    tx, ty = PLACE_TRIM[unit]
    return L.LINE_X + tx, L.SLOTS[slot] + ty


def turn(unit: str) -> float:
    """The quarter turn (rad) that brings this unit's battered face round to the road (heading -90 degrees)."""
    return math.remainder(L.LAID_HEADING - L.UNIT_START[unit][2], 2 * math.pi)


class _Clock:
    """One arm's waypoints with a running clock, so the two arms' plans can be lined up in time."""

    def __init__(self) -> None:
        self.wps: list[Waypoint] = []
        self.t = 0.0

    def add(self, duration, pos, finger, label, yaw) -> None:
        self.wps.append(Waypoint(duration, pos, finger, label, yaw))
        self.t += duration

    def hold_until(self, t: float, label: str) -> None:
        """Stay where the last waypoint left the arm until time t."""
        if self.wps and t > self.t + 1e-9:
            w = self.wps[-1]
            self.add(t - self.t, w.pos, w.finger, label, w.yaw)


def _pick(c: _Clock, unit: str, start: float) -> None:
    """Over the unit, down (slowly for the last 30 mm), grip, lift."""
    sx, sy, _ = L.UNIT_START[unit]
    c.hold_until(start, "wait")
    c.add(2.0 if not c.wps else 1.5, (sx, sy, HOVER), OPEN, "over the unit", 0.0)
    c.add(1.0, (sx, sy, GRIP_Z + APPROACH), OPEN, "down to just above the unit", 0.0)
    c.add(1.5, (sx, sy, GRIP_Z), OPEN, "slowly down round the unit", 0.0)
    c.add(0.5, (sx, sy, GRIP_Z), OPEN, "settle", 0.0)
    c.add(1.5, (sx, sy, GRIP_Z), GRIP, "grip the unit", 0.0)
    c.add(0.3, (sx, sy, GRIP_Z), GRIP, "settle the grip", 0.0)
    c.add(1.2, (sx, sy, HOVER), GRIP, "lift the unit", 0.0)


def _place(c: _Clock, unit: str, slot: str, start: float) -> None:
    """Carry to the slot turning the unit's face round to the road, down (slowly), let go, up."""
    c.hold_until(start, "hold the unit until the kerb line is free")
    dx, dy = place_point(unit, slot)
    yaw = turn(unit)
    place_z = GRIP_Z + DROP
    c.add(2.0, (dx, dy, HOVER), GRIP, "carry it to the slot, turning its face to the road", yaw)
    c.add(0.5, (dx, dy, HOVER), GRIP, "settle over the slot", yaw)
    c.add(1.0, (dx, dy, place_z + APPROACH), GRIP, "down to just above the bed", yaw)
    c.add(1.5, (dx, dy, place_z), GRIP, "slowly set the unit down", yaw)
    c.add(0.3, (dx, dy, place_z), GRIP, "settle", yaw)
    c.add(1.5, (dx, dy, place_z), OPEN, "let go of the unit", yaw)
    c.add(1.0, (dx, dy, HOVER), OPEN, "straight up", yaw)


def _arm(side: str) -> list[Waypoint]:
    """One arm's script, on the shared clock."""
    c = _Clock()
    k = SIDE[side]
    unit = None
    for kind, what, start in SCRIPT[side]:
        if kind == "pick":
            unit = what
            _pick(c, unit, start)
        elif kind == "place":
            _place(c, unit, what, start)
        else:  # park
            c.hold_until(start, "wait")
            c.add(2.0 if not c.wps else 1.5, (PARK[0], k * PARK[1], PARK[2]), OPEN, "park beside the own base",
                  PARK_YAW[side])
    return c.wps


def demo():
    """Both arms on one clock, taking turns at the kerb line (S1 left, S3 right, S2 left, S4 right); each reaches
    across the midline for its transition only while the other is parked beside its own base. DEMO_HOLD_ONLY=1: no
    plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped
    (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
