"""On the station table, a floor plan of an accessible toilet room at 1 in = 4 mm: a grey kerb for the rear wall
(face at x 45.5 cm, from y -32.5 to +0.5 cm) and one for the side wall (face at y -31.5 cm, from x 20.5 to 45.5 cm),
and the toilet painted on the floor (a round bowl and a tank against the rear wall, 18 in wide, its centreline 17 in
from the side wall, at y -24.7 cm). Two loose grab bars (20 x 20 mm aluminium tube, 48 in = 192 mm and 40 in =
160 mm long) lie in the room, the 48 in one parallel to the rear wall and the 40 in one parallel to the side wall.
Rule (2010 ADA Standards 604.5): a bar along each wall; the side-wall bar at least 42 in long, its rear end at most
12 in from the rear wall and its front end at least 54 in from it; the rear-wall bar at least 36 in long, reaching at
least 12 in past the centreline toward the side wall and 24 in toward the open side. Only the 48 in bar can be the
side bar, and the rear bar must sit off centre toward the open side.

The scorer also reads the arms (TCP, pad-tip point, opening) through rigstate.py, which wraps the embodiment's step
and reset from this file's import; it changes nothing else the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the demo uses only the right arm; DEMO_IDLE_ARM=right leaves it at its ready pose instead.
"""

from __future__ import annotations

import json
import math
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
# Full credit already needs both bars let go for 0.5 s and still for 1 s; the trial then ends once that has held a
# further 1 s.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "This floor plan of an accessible toilet room is drawn at 1 in = 4 mm. The grey kerbs are the rear wall behind "
    "the toilet and the side wall beside it; the white shapes are the toilet's bowl and tank. Fit the two grab bars "
    "(40 in and 48 in long), one on each wall, where the 2010 ADA Standards require them for this toilet. Lay each "
    "bar flat on the floor along its wall, about 1-1/2 in off the kerb, then let go of both."
)


def _span_box(name, x0, x1, y0, y1, h, color) -> Box:
    """A fixed box on the table given by its extents (x, y in the world) and height."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + h / 2), size=(x1 - x0, y1 - y0, h), color=color)


REAR_KERB = _span_box("rear_wall_kerb", *L.REAR_KERB, L.KERB_H, L.KERB_GREY)
SIDE_KERB = _span_box("side_wall_kerb", *L.SIDE_KERB, L.KERB_H, L.KERB_GREY)
TANK = _span_box("toilet_tank_paint", *L.TANK, L.PAINT_T, L.TANK_WHITE)
BOWL = Cylinder("toilet_bowl_paint", pos=(L.BOWL_X, L.CL_Y, L.TOP + L.PAINT_T / 2), radius=L.BOWL_R,
                height=L.PAINT_T, color=L.PAINT_WHITE)
BARS = [
    Box(name, pos=(x, y, L.TOP + L.BAR_W / 2), size=(L.BAR_LEN[name], L.BAR_W, L.BAR_W), mass=L.BAR_MASS[name],
        color=L.STEEL, yaw=math.radians(yaw))
    for name, (x, y, yaw) in L.BAR_START.items()
]
OBJECTS = [REAR_KERB, SIDE_KERB, TANK, BOWL, *BARS]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (a teleport of both bars back to the start is another)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided. A trace
    that cannot be written is skipped (it must never break a run)."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def _bars(objects) -> list:
    return [objects[n] for n in L.NAMES]


def success(objects) -> bool:
    """Both walls compliant (the 48 in bar along the side wall within 12 in of the rear wall and reaching 54 in out,
    the 40 in bar along the rear wall reaching 12 in past the centreline toward the side wall and 24 in toward the
    open side), each bar flat on the floor, let go for 0.5 s and still for 1 s; held 1 s. This runs first every
    control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    result = _TRACKER.update(_bars(objects), arms)
    row = {"episode": rigstate.episode(), "score": result.score, "success": result.success, **result.details}
    if arms is not None:
        try:
            row.update(right_tcp=arms["right"].tcp, right_tips=arms["right"].tips, left_tips=arms["left"].tips,
                       openings=[arms["left"].opening, arms["right"].opening])
        except (KeyError, AttributeError):
            row["rigstate_error"] = "incomplete snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 both walls compliant and both bars still for 1 s; 0.75 one wall compliant and a bar along the other;
    0.5 one wall compliant and the other bare; 0.25 a bar along a wall, neither compliant; 0 no bar along a wall.
    A bar counts only flat on the floor, let go (finger zones 10 mm off) for 0.5 s and settled. Read from the state
    at hand (the last step)."""
    return _TRACKER.result(_bars(objects), rigstate.arms()).score


# -- the scripted demo (right arm only: both walls are beside its base) -------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.017  # 34 mm: 7 mm either side of a 20 mm bar
GRIP = -(L.BAR_W - 0.008) / 2  # 12 mm commanded across the bar's 20 mm
PAD = 0.013  # the pad tips are this far below the tool point
GRIP_UP = 0.008 + PAD  # the TCP over the bar's underside: pad tips 8 mm above it, holding the upper 12 mm
DROP = 0.002  # each bar is let go with its underside this far above the floor
APPROACH = 0.030  # each final descent starts this far above its end point and takes 1.5 s
HOVER = L.TOP + 0.100  # carried with the bar's underside 79 mm up
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.TOP + DROP + GRIP_UP
# Grip headings. Yaw 0 closes the fingers along y. Each bar is gripped at its middle across its 20 mm width and
# turned a quarter turn while carried: the 48 in bar (along y) at yaw +90, laid along x at yaw 0; the 40 in bar
# (along x) at yaw 0, laid along y at yaw +90. In our GPU runs parts carried in a gripper ended turned +0.6 to +1.7
# degrees counter-clockwise of the commanded yaw, so each place yaw is set 1.15 degrees clockwise (offline, with no
# such turn, the bars lie 1.15 degrees turned, well inside the 5 degree limit).
YAW_BIAS = math.radians(-1.15)
YAW = {"bar_48": (math.radians(90.0), 0.0 + YAW_BIAS), "bar_40": (0.0, math.radians(90.0) + YAW_BIAS)}
# Trim (x, y) of each place point: in our GPU runs parts landed about 4 mm short of the place point along the
# carry, so each aim is moved 4 mm further along its carry direction. If a bar lands off its aim (TASK_TRACE:
# centre_mm of bar_48 away from (321, -297), or of bar_40 away from (437, -223), by more than 3 mm), move its
# trim by minus the offset. The windows: the side bar's centre x 311..335 mm and y -306..-290 mm; the rear bar's
# centre x 430..446 mm and y -231..-215 mm.
CARRY_SHORT = 0.004


def _trim(name: str) -> tuple[float, float]:
    sx, sy, _ = L.BAR_START[name]
    tx, ty = L.TARGET[name]
    d = math.hypot(tx - sx, ty - sy)
    return CARRY_SHORT * (tx - sx) / d, CARRY_SHORT * (ty - sy) / d


PLACE_TRIM = {n: _trim(n) for n in L.NAMES}  # bar_48 (+1.7, -3.6) mm; bar_40 (+1.9, -3.5) mm


def place_point(name: str) -> tuple[float, float]:
    tx, ty = L.TARGET[name]
    dx, dy = PLACE_TRIM[name]
    return tx + dx, ty + dy


REST = (0.200, -0.400, L.TOP + 0.150)  # where the right arm waits at the end: back by its base, clear of both bars


def _fit(name: str, last: bool) -> list[Waypoint]:
    """Over the bar's middle, down (slowly for the last 30 mm), grip it across its width, lift, carry it beside its
    wall turning it a quarter turn, down (slowly), let go, straight up; after the last bar, out of the way."""
    sx, sy, _ = L.BAR_START[name]
    px, py = place_point(name)
    yaw_pick, yaw_place = YAW[name]
    plan = [
        Waypoint(2.0, (sx, sy, HOVER), OPEN, f"over the {name} middle", yaw_pick),
        Waypoint(1.0, (sx, sy, PICK_Z + APPROACH), OPEN, "down to just above it", yaw_pick),
        Waypoint(1.5, (sx, sy, PICK_Z), OPEN, "slowly down round it", yaw_pick),
        Waypoint(0.5, (sx, sy, PICK_Z), OPEN, "settle", yaw_pick),
        Waypoint(1.5, (sx, sy, PICK_Z), GRIP, "grip it across its width", yaw_pick),
        Waypoint(0.3, (sx, sy, PICK_Z), GRIP, "settle the grip", yaw_pick),
        Waypoint(1.2, (sx, sy, HOVER), GRIP, "lift it", yaw_pick),
        Waypoint(2.5, (px, py, HOVER), GRIP, "carry it beside its wall, turning it a quarter turn", yaw_place),
        Waypoint(0.5, (px, py, HOVER), GRIP, "settle over the place", yaw_place),
        Waypoint(1.0, (px, py, PLACE_Z + APPROACH), GRIP, "down to just above the floor", yaw_place),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly lay it down", yaw_place),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", yaw_place),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", yaw_place),
        Waypoint(1.0, (px, py, HOVER), OPEN, "straight up", yaw_place),
    ]
    if last:
        plan.append(Waypoint(1.5, REST, OPEN, "out of the way", 0.0))  # yaw 0 there: joint 6 near mid-range
    return plan


def demo():
    """The right arm fits the 48 in bar to the side wall, then the 40 in bar to the rear wall; the left arm holds
    its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"right": [*_fit("bar_48", last=False), *_fit("bar_40", last=True)]}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
