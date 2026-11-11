"""On the station table, at 1:10 scale: a classroom with a grey vinyl floor (310 x 560 mm) and, at its far end, the
front wall (cut down to 60 mm) with a whiteboard and the exam clock on it. Four single desks (60 x 45 mm tops, 72 mm
tall, each with a blue chair fixed behind it; one 80 g rigid unit each, loose) stand as a seminar left them: two rows
facing each other across the middle of the room, each row's desks 100 mm apart centre to centre. Rule (UK exam
seating, the JCQ's Instructions for Conducting Examinations: all facing the same way, here the usual way, the front):
every candidate faces the front (the whiteboard and the clock), each desk square to the front wall within 15
degrees, and no two desks closer than 1.25 m (125 mm here) centre to centre in any direction.

Each desk is a mechkit compound (mechkit.py, next to this file: one loose rigid body of eight boxes), so its chair
shows which way the candidate faces and the scorer can read it. It changes nothing else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the desks are built as compound bodies)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, all four desks set, facing the front and spaced, still) must hold 3 s, not the default 1 s. A desk
# counts as set once its lowest point is within 3 mm of the floor, which happens at the end of the slow descent while
# the grippers still hold the last two desks; settling, letting go and rising take 0.6 + 1.5 + 1.5 s, so the trial
# ends with the fingers open, going up.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Set this classroom out for a written exam (1:10 scale). Move and turn the four desks, each with its chair fixed "
    "behind it, so that every candidate would face the front of the room (the whiteboard and the clock), each desk "
    "square to the front wall within 15 degrees, and no two desks are closer than 125 mm (1.25 m) centre to centre "
    "in any direction, measured between the middles of the desk tops. Leave every desk standing upright on the grey "
    "floor."
)


def _span_box(name, xs, ys, zs, colour, **kw) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    (x0, x1), (y0, y1), (z0, z1) = xs, ys, zs
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=L.COLOURS[colour], **kw)


FLOOR = _span_box("room_floor", L.ROOM_X, L.ROOM_Y, (0.0, L.FLOOR_T), "floor")
FRONT_WALL = _span_box("front_wall", *L.WALL, "wall")
WHITEBOARD = _span_box("whiteboard", *L.WHITEBOARD, "whiteboard")
CLOCK = [_span_box("exam_clock", *L.CLOCK_RIM, "clock_rim"), _span_box("clock_face", *L.CLOCK_FACE, "clock_face")]
ROOM = [FLOOR, FRONT_WALL, WHITEBOARD, *CLOCK]


def _desk(name: str):
    x, y, heading = L.START[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP + L.FLOOR),
        yaw=heading,
        mass=L.DESK_MASS,
        color=L.COLOURS["beech"],
        shapes=[mk.box(c, s, color=L.COLOURS[k]) for _, c, s, k in L.DESK_SHAPES],
    )


DESK_OBJECTS = [_desk(n) for n in L.DESKS]
OBJECTS = [*ROOM, *DESK_OBJECTS]

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


def _desks(objects) -> dict:
    return {n: objects.get(n) for n in L.DESKS}


def success(objects) -> bool:
    """All four desks credited (score 1.0): each set (upright, on the floor, its desk-top centre over the floor,
    still), facing the front within 15 degrees and at least 125 mm from every other desk, centre to centre; held 3 s.
    This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    ds = _desks(objects)
    result = _TRACKER.update(ds)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(ds[L.DESKS[0]], "source", None),
                   origins={n: d.origin for n, d in ds.items()},
                   quats={n: d.quat_xyzw for n, d in ds.items()},
                   low_z={n: d.low[2] for n, d in ds.items()},
                   vels={n: d.lin_vel for n, d in ds.items()})
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each desk that is set (tilted 10 degrees or less, lowest point within 3 mm of the floor, its desk-top
    centre over the grey floor, still over 0.5 s), faces the front (its chair-to-desk direction within 15 degrees of
    square to the front wall) and is spaced (its desk-top centre at least 125 mm, seen from above, from every other
    desk's, set or not). Read from the state at hand (the last step)."""
    return _TRACKER.result(_desks(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Both arms at once, each turning and moving the two desks on its own side of the room: the left arm turns desks 1
# and 2 a quarter turn anticlockwise (facing -y to facing +x), the right arm turns desks 3 and 4 a quarter turn
# clockwise (facing +y to +x); none is turned the other way, which would leave it facing the back. Each grip is across
# the desk's two side frames (the fingers close along its 60 mm width) and the turn is made with the wrist while the
# desk is carried, so its heading changes with the grip's yaw. Yaw 0 closes the fingers along world y, so a desk
# facing h is held at yaw h: the left arm from -90 to 0 degrees, the right arm from +90 to 0, each turn in two
# segments (the README's rule of thumb: the left arm reaches across with yaw -90, the right with +90).
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
APPROACH_OPEN = -0.038  # 76 mm: down beside a desk's side frames, 8 mm clear each side of its 60 mm width
RELEASE_OPEN = APPROACH_OPEN  # let go to the same opening
GRIP = -(0.060 - 0.008) / 2  # 52 mm commanded on the 60 mm frames
GRIP_U = -0.010  # the grip 10 mm behind the middle of the desk, toward the chair: within 4 mm of the centre of mass
# The TCP 60 mm above the table: the pads (from 25 mm above the TCP to their tips 13 mm below it, 20 mm wide) on the
# side frames' outer faces and the desk top's edges from 47 mm up to the top (73 mm), all within the frames' 45 mm
# depth. Above the pads, the finger carriages (57 mm above the TCP, out to +-71 mm along the closing axis when open)
# and the housing (about 70 mm above the TCP) pass 30 mm or more over the tallest part of any desk (the chair backs,
# 87 mm), so nothing but the pads touches a desk.
GRIP_Z = L.TOP + 0.060
LIFT = 0.020  # the desk lifted 20 mm to carry it (nothing on the floor is higher than the 1 mm floor's edge)
APPROACH = 0.030  # the last 30 mm down to the grip are slow
PLACE_UP = 0.002  # let go with the desk 2 mm above the floor
ABOVE_Z = L.TOP + 0.120  # pad tips 107 mm up, 20 mm over the chair backs (87 mm)
PARK = {"left": (0.250, 0.290, L.TOP + 0.160), "right": (0.250, -0.290, L.TOP + 0.160)}
SIDE = {"left": ("desk_1", "desk_2"), "right": ("desk_3", "desk_4")}


def grip_point(origin_xy, heading: float) -> tuple[float, float]:
    """Where the grip is (world x, y) for a desk whose origin is at ``origin_xy`` facing ``heading``."""
    return (origin_xy[0] + GRIP_U * math.cos(heading), origin_xy[1] + GRIP_U * math.sin(heading))


def _yaw(arm: str, heading: float) -> float:
    """The grip yaw for a desk facing ``heading`` (either arm: the fingers close across its width)."""
    return math.remainder(heading, 2 * math.pi)


def _cycle(arm: str, name: str, came_from=None) -> list[Waypoint]:
    """One desk: over it (from the ready pose, or from over the place ``came_from`` turning the wrist back), down
    beside its side frames, slowly for the last 30 mm, grip across them, lift 20 mm, carry it to its place turning it
    (two segments), lower it slowly, let go, straight up."""
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
        Waypoint(1.5, (sx, sy, approach_z), APPROACH_OPEN, "down beside its side frames", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), APPROACH_OPEN, "down the last 30 mm slowly", ys),
        Waypoint(0.5, (sx, sy, GRIP_Z), APPROACH_OPEN, "settle", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, "grip across the side frames", ys),
        Waypoint(0.3, (sx, sy, GRIP_Z), GRIP, "settle the grip", ys),
        Waypoint(1.5, (sx, sy, carry_z), GRIP, "lift 20 mm", ys),
        Waypoint(2.0, (mx, my, carry_z), GRIP, "carry it, turning it", ym),
        Waypoint(2.0, (px, py, carry_z), GRIP, "carry it to its place, finishing the turn", yp),
        Waypoint(0.5, (px, py, carry_z), GRIP, "settle over the place", yp),
        Waypoint(2.0, (px, py, place_z), GRIP, "lower it slowly", yp),
        Waypoint(0.6, (px, py, place_z), GRIP, "settle", yp),
        Waypoint(1.5, (px, py, place_z), RELEASE_OPEN, "let go", yp),
        Waypoint(1.5, (px, py, ABOVE_Z), RELEASE_OPEN, "straight up", yp),
    ]


def _arm(arm: str) -> list[Waypoint]:
    """One arm: the desk nearer the arms to the place nearer the arms, then the other to the place near the front
    wall, then out of the way."""
    near, far = SIDE[arm]
    first = _cycle(arm, near)
    return [*first, *_cycle(arm, far, came_from=first[-1].pos[:2]),
            Waypoint(2.0, PARK[arm], RELEASE_OPEN, "out of the way", _yaw(arm, L.TARGET[far][2]))]


def demo():
    """Both arms on one clock, each turning and moving the two desks on its own side. DEMO_HOLD_ONLY=1: no plan (both
    arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm
    check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
