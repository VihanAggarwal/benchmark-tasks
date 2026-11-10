"""On the station table, at 1:16 scale: a wet room with a mid-grey carpet (felt, 350 x 370 mm), a left wall, a right
wall and a back wall cut down to their bottom 10 mm (16 cm), and no wall on the side nearest the arms. Four blue air
movers (30 x 26 x 30 mm bodies, black over their front 5 mm, with a black outlet at the bottom of the front and a grey
handle on top, 25 g each, loose) stand in the middle of the room, two on each side, all blowing toward the right wall. Rule (water-damage
restoration): two air movers along each side wall, each with its outlet within 15 mm (24 cm) of the wall and blowing
into the wall at 15 to 45 degrees so the air sweeps along it, the two on each wall 190 to 305 mm (10 to 16 ft) apart,
and all four blowing the same way round the room (either way).

Each mover is a mechkit compound (mechkit.py, next to this file: one loose rigid body of four boxes), so its front
and outlet can be seen and the scorer can read which way it blows. It changes nothing else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the movers are built as compound bodies)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, all four movers set and still) must hold 3 s, not the default 1 s. A mover counts as placed once its
# lowest point is within 3 mm of the carpet, which happens at the end of the slow descent while the grippers still
# hold the last two movers; settling, letting go and rising take 0.6 + 1.5 + 1.5 s, so the trial ends with the
# fingers open, going up.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "The carpet in this room is wet (1:16 scale; the wall nearest you is left out). Set the four air movers to dry it "
    "the way restoration crews do: two along each side wall, each with its black outlet no more than 15 mm (24 cm) "
    "from the wall and blowing into the wall at 15 to 45 degrees so the air sweeps along it, the two on each wall 190 "
    "to 305 mm (10 to 16 ft) apart, and all four blowing the same way round the room."
)


def _span_box(name, xs, ys, zs, colour, **kw) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    (x0, x1), (y0, y1), (z0, z1) = xs, ys, zs
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=L.COLOURS[colour], **kw)


CARPET = _span_box("carpet", L.ROOM_X, L.ROOM_Y, (0.0, L.CARPET_T), "carpet")
WALL_BOXES = [_span_box(name, xs, ys, (0.0, L.WALL_H), "wall") for name, xs, ys in L.WALLS]
ROOM = [CARPET, *WALL_BOXES]


def _mover(name: str):
    x, y, heading = L.START[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP + L.FLOOR),
        yaw=heading,
        mass=L.MOVER_MASS,
        color=L.COLOURS["mover_blue"],
        shapes=[mk.box(c, s, color=L.COLOURS[k]) for _, c, s, k in L.MOVER_SHAPES],
    )


MOVER_OBJECTS = [_mover(n) for n in L.MOVERS]
OBJECTS = [*ROOM, *MOVER_OBJECTS]

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


def _movers(objects) -> dict:
    return {n: objects.get(n) for n in L.MOVERS}


def success(objects) -> bool:
    """All four movers set (score 1.0): each placed (upright, on the carpet, inside the room, still) with its outlet
    within 15 mm of the side wall it blows into, at 15 to 45 degrees to it; two on each wall, 190-305 mm apart; all
    four the same way round the room; held 3 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    ms = _movers(objects)
    result = _TRACKER.update(ms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(ms[L.MOVERS[0]], "source", None),
                   origins={n: m.origin for n, m in ms.items()},
                   quats={n: m.quat_xyzw for n, m in ms.items()},
                   low_z={n: m.low[2] for n, m in ms.items()},
                   vels={n: m.lin_vel for n, m in ms.items()})
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each credited air mover, counted for whichever way round the room credits more. A mover is set when it
    is placed (tilted 10 degrees or less, lowest point within 3 mm of the carpet, centre inside the room, still over
    0.5 s) and blows into a side wall at 15 to 45 degrees with its outlet no more than 15 mm from that wall's face; it
    blows clockwise seen from above if its airflow points away from the arms along the left wall or toward them along
    the right wall. A wall credits at most two movers of one sense, and two only if their centres are 190-305 mm apart
    along it (else one). Read from the state at hand (the last step)."""
    return _TRACKER.result(_movers(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Both arms at once, each setting the two movers on its own side along its own wall, clockwise seen from above: the
# left arm turns each of its movers 120 degrees anticlockwise (airflow heading -90 to +30 degrees: away from the arms,
# 30 degrees into the left wall), the right arm turns each of its movers 60 degrees clockwise (heading -90 to -150
# degrees: toward the arms, 30 degrees into the right wall). Each grip is across the mover's 26 mm body (the fingers
# close along its width) and the turn is made with the wrist while the mover is carried, so its heading changes with
# the grip's yaw. Yaw 0 closes the fingers along world y, so a mover heading h is held at yaw h or h + 180 degrees:
# the left arm holds its movers at yaw h (-90 at the grip, +30 at the place; joint 6 from about +62/+67 to -68/-50
# degrees), the right arm at yaw h + 180 (+90 to +30; joint 6 from about -62/-67 to +8/-11 degrees), each turn in two
# segments. Joint 6 stays inside its +-120 degrees; the grips 180 degrees round would not.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
APPROACH_OPEN = -0.030  # 60 mm: down beside a mover (17 mm clear each side of the 26 mm body)
RELEASE_OPEN = -0.025  # 50 mm: let go (12 mm clear each side), so the wall-side pad stays near the wall
GRIP = -(0.026 - 0.008) / 2  # 18 mm commanded on the 26 mm body
GRIP_U = 0.0  # the grip at the middle of the body (the centre of mass is 1 mm toward the outlet)
# The TCP 30 mm above the table: the pads (from 25 mm above the TCP to their tips 13 mm below it) on the body from 17 mm
# up to its top (31 mm), the tips 7 mm over the 10 mm walls, so the wall-side pad (5.8 mm from the wall's face at the
# place) never meets a wall; the 10 mm handle sits between the pads, 8 mm clear of each; the gripper's body (35 mm above
# the TCP) is 28 mm over the handle.
GRIP_Z = L.TOP + 0.030
LIFT = 0.030  # the mover lifted 30 mm to carry it: its base 31 mm up, 21 mm over the walls (no carry passes a mover)
APPROACH = 0.030  # the last 30 mm down to the grip are slow
PLACE_UP = 0.002  # let go with the mover's base 2 mm above the carpet
ABOVE_Z = L.TOP + 0.100  # pad tips 87 mm up, 50 mm over the handles
PARK = {"left": (0.200, 0.300, L.TOP + 0.150), "right": (0.200, -0.300, L.TOP + 0.150)}
SIDE = {"left": ("air_mover_1", "air_mover_2"), "right": ("air_mover_3", "air_mover_4")}
# Grip yaw minus the mover's heading: 0 for the left arm, 180 degrees for the right (joint 6 stays inside +-120).
YAW_OFFSET = {"left": 0.0, "right": math.pi}


def grip_point(origin_xy, heading: float) -> tuple[float, float]:
    """Where the grip is (world x, y) for a mover whose origin is at ``origin_xy`` heading ``heading``."""
    return (origin_xy[0] + GRIP_U * math.cos(heading), origin_xy[1] + GRIP_U * math.sin(heading))


def _yaw(arm: str, heading: float) -> float:
    """The grip yaw for a mover heading ``heading``, in the arm's working range (-180 .. 180 degrees)."""
    return math.remainder(heading + YAW_OFFSET[arm], 2 * math.pi)


def _cycle(arm: str, name: str, came_from=None) -> list[Waypoint]:
    """One mover: over it (from the ready pose, or from over the place ``came_from`` turning the wrist back), down
    slowly, grip across its width, lift 30 mm, carry it to its place turning it (two segments), lower it slowly, let
    go, straight up."""
    x0, y0, h0 = L.START[name]
    x1, y1, h1 = L.TARGET[name]
    sx, sy = grip_point((x0, y0), h0)
    px, py = grip_point((x1, y1), h1)
    mx, my = (sx + px) / 2, (sy + py) / 2
    ys, yp = _yaw(arm, h0), _yaw(arm, h1)
    ym = ys + math.remainder(yp - ys, 2 * math.pi) / 2
    carry_z, place_z, approach_z = GRIP_Z + LIFT, GRIP_Z + PLACE_UP, GRIP_Z + APPROACH
    label = name.replace("_", " ")
    if came_from is None:
        head = [Waypoint(3.0, (sx, sy, ABOVE_Z), APPROACH_OPEN, f"over {label}", ys)]
    else:  # from over the last place, turning the wrist back in two segments (no segment over 90 degrees)
        cx, cy = came_from
        head = [
            Waypoint(2.0, ((sx + cx) / 2, (sy + cy) / 2, ABOVE_Z), APPROACH_OPEN, f"toward {label}", ym),
            Waypoint(1.0, (sx, sy, ABOVE_Z), APPROACH_OPEN, f"over {label}", ys),
        ]
    return head + [
        Waypoint(1.5, (sx, sy, approach_z), APPROACH_OPEN, "down beside its body", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), APPROACH_OPEN, "down the last 30 mm slowly", ys),
        Waypoint(0.5, (sx, sy, GRIP_Z), APPROACH_OPEN, "settle", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, "grip across the body", ys),
        Waypoint(1.5, (sx, sy, carry_z), GRIP, "lift 30 mm", ys),
        Waypoint(2.0, (mx, my, carry_z), GRIP, "carry it, turning it", ym),
        Waypoint(2.0, (px, py, carry_z), GRIP, "carry it to its place, finishing the turn", yp),
        Waypoint(0.5, (px, py, carry_z), GRIP, "settle over the place", yp),
        Waypoint(2.0, (px, py, place_z), GRIP, "lower it slowly", yp),
        Waypoint(0.6, (px, py, place_z), GRIP, "settle", yp),
        Waypoint(1.5, (px, py, place_z), RELEASE_OPEN, "let go", yp),
        Waypoint(1.5, (px, py, ABOVE_Z), RELEASE_OPEN, "straight up", yp),
    ]


def _arm(arm: str) -> list[Waypoint]:
    """One arm: the mover nearer the arms to the place nearer the arms, then the other to the place by the back wall,
    then out of the way."""
    near, far = SIDE[arm]
    first = _cycle(arm, near)
    return [*first, *_cycle(arm, far, came_from=first[-1].pos[:2]),
            Waypoint(2.0, PARK[arm], RELEASE_OPEN, "out of the way", _yaw(arm, L.TARGET[far][2]))]


def demo():
    """Both arms on one clock, each setting the two movers on its own side. DEMO_HOLD_ONLY=1: no plan (both arms hold
    the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
