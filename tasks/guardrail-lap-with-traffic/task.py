"""On the station table: a bench model of a W-beam guardrail run along the far edge of a road. Three dark brown posts
(25 x 25 x 60 mm, fixed) stand in a row along y at x 32.25 cm, 24 cm apart (y +24, 0 and -24 cm), their road-side
faces at x 31.0 cm. In front of them (toward the arms) lies a strip of dark grey road with a white edge line and
three white arrows pointing to the robot's left (+y): traffic runs from the robot's right to its left. Two loose
grey rail panels (pine battens 300 x 18 x 45 mm, 0.12 kg) stand on edge outside the posts, lengths pointing away
from the robot, centred at x 31 cm, y +40 and -40 cm. Rule: splice them at the middle post, lapped in the direction
of traffic. One panel (downstream, D) stands against the left and middle posts, its right end 0-55 mm past the
middle post's centre; the other (upstream, U) stands against the right post with its left end in front of
(nearer the road than) D's right end, overlapping it 30-60 mm, centred on the middle post. A lap the other way
round (U's end behind D's) scores 0. Then both grippers let go and lift clear (fingers 10 mm off both panels).

The scorer reads both grippers (tool point, finger-tip point, opening) through rigstate.py, which wraps the
embodiment's step and reset from this file's import; it changes nothing the embodiment does.

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
import rigstate  # noqa: E402
from scoring import PANELS, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 needs both grippers' finger zones 10 mm clear of both panels, so the hold can only start once the second
# gripper has let go and lifted: 1 s of it ends the trial with both hands clear.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Splice the two grey guardrail panels at the middle post. Stand one on edge against the left and middle posts "
    "and the other on edge against the right post, their ends overlapping 3 to 6 cm, centred on the middle post. "
    "Lap them in the direction of traffic, which runs the way the white arrows point: at the splice, the end of the "
    "panel that traffic reaches first must lie on the road side of (nearer you than) the other panel's end. Then "
    "let go of both panels and lift both grippers clear."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color, **kw)


POST_OBJECTS = [
    Box(name, pos=(L.POST_X, y, L.TOP + L.POST_SIZE[2] / 2), size=L.POST_SIZE, color=L.POST_BROWN)
    for name, y in L.POSTS.items()
]
ROAD = _span_box("road", L.ROAD_X0, L.ROAD_X1, -L.ROAD_HALF_Y, L.ROAD_HALF_Y, 0.0, L.ROAD_T, L.ROAD_GREY)
EDGE_LINE = _span_box("edge_line", L.LINE_X - L.LINE_W / 2, L.LINE_X + L.LINE_W / 2, -L.ROAD_HALF_Y, L.ROAD_HALF_Y,
                      L.ROAD_T, L.ROAD_T + L.MARK_T, L.WHITE)
_MARK_Z = L.TOP + L.ROAD_T + L.MARK_T / 2


def _arrow(i: int, yc: float) -> list[Box]:
    """A traffic arrow pointing +y: a shaft along y and two head strokes meeting at the tip."""
    shaft = Box(f"arrow{i}_shaft", pos=(L.ARROW_X, yc + (L.SHAFT_Y0 + L.SHAFT_Y1) / 2, _MARK_Z),
                size=(L.ARROW_W, L.SHAFT_Y1 - L.SHAFT_Y0, L.MARK_T), color=L.WHITE)
    tip = yc + L.ARROW_TIP
    heads = [
        Box(f"arrow{i}_head_{side}", pos=(L.ARROW_X + sgn * L.HEAD_DX, tip - L.HEAD_DY, _MARK_Z),
            size=(L.ARROW_W, L.HEAD_LEN, L.MARK_T), color=L.WHITE, yaw=sgn * math.pi / 4)
        for side, sgn in (("a", -1.0), ("b", 1.0))
    ]
    return [shaft, *heads]


ARROWS = [b for i, yc in enumerate(L.ARROW_YS) for b in _arrow(i, yc)]
PANEL_OBJECTS = [
    Box(name, pos=(x, y, L.TOP + L.PANEL_HALF[2]), size=L.PANEL_SIZE, mass=L.PANEL_MASS, color=L.GALV_GREY,
        yaw=L.PANEL_START_YAW)
    for name, (x, y) in L.PANEL_START.items()
]
OBJECTS = [*POST_OBJECTS, ROAD, EDGE_LINE, *ARROWS, *PANEL_OBJECTS]

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


def _panels(objects) -> dict:
    return {n: objects.get(n) for n in PANELS}


def success(objects) -> bool:
    """D standing against the left and middle posts (back edge 0-8 mm in front of their faces, right end 0-55 mm
    past the middle post's centre), U standing in front of D's end (lap gap -1..15 mm all along the 30-60 mm
    overlap, centred within 15 mm of the middle post, turned under 15 degrees) and against the right post within
    12 mm, both still, both grippers' finger zones 10 mm clear; held 1 s. This runs first every control step, so
    it also feeds the tracker (with the arms rigstate read for this step)."""
    panels = _panels(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(panels, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in panels.items():
        if s is not None:
            row[f"{n}_pos"] = getattr(s, "pos", None)
            row[f"{n}_quat"] = getattr(s, "quat_xyzw", None)
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_opening=arms["left"].opening, right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if no panel stands in the downstream place (against the left and middle posts, right end 0-55 mm past
    the middle post's centre), or if the other panel's end is tucked behind its end (the wrong way). Else 1.0 if the
    other panel is lapped on the road side and the splice is right (overlap 30-60 mm, centred within 15 mm, against
    the right post within 12 mm) with both grippers' finger zones 10 mm clear; 0.5 if it is lapped on the road side
    but not all of that; 0.25 otherwise. 0, 0.25, 0.5 or 1.0."""
    return _TRACKER.result(_panels(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
RELEASE = -0.0435  # 87 mm: 34.5 mm clear of each face of the 18 mm panel
GRIP = -(L.PANEL_SIZE[0] - 0.008) / 2  # -0.005: across the panel's 18 mm thickness, 10 mm commanded
PICK_Z = L.TOP + 0.032  # TCP 32 mm above the panel's bottom: the pads (tips 19 mm up) grip its top 26 mm
MID_Z = L.TOP + 0.060  # slows the last 28 mm of the descent onto the panel
CARRY_Z = L.TOP + 0.120  # carried with the panel's bottom 88 mm up: 28 mm over the posts' tops
ENTRY_Z = L.TOP + 0.050  # the panel's bottom 18 mm up: below the posts' tops, in front of them
SLIDE_Z = L.TOP + 0.040  # moved sideways into place with the panel's bottom 8 mm up (about 5 mm with the arm's sag)
PLACE_Z = L.TOP + 0.0335  # then straight down: let go with its bottom 1.5 mm up (the arm sags about 3 mm under it)
LIFT_Z = L.TOP + 0.100  # the open hand straight up: its tool point 55 mm over the panel's top
PARK_Z = L.TOP + 0.150
# Downstream panel D (left arm, fingers closing along x: yaw -pi/2). Its centre at y 0.125 puts its right end at
# y -0.025, 12.5 mm past the middle post's right side. It comes down 11 mm in front of the posts, is pressed back into
# them with its bottom clear of the table, then set straight down: the place point puts its back face 2 mm past
# their faces (x 0.312), so the compliant arm pushes it flush.
D_Y = 0.125
D_ENTRY_X = 0.290  # back face at x 0.299
D_PLACE_X = L.POST_FACE_X - L.PANEL_HALF[0] + 0.002  # 0.303
# Upstream panel U (right arm). Rigid battens do not nest, so U stands free in front of D at the splice, turned
# 4.72 deg anticlockwise (its right end farther from the arms): its back-left corner (x 0.286, y +0.020) is 6 mm in
# front of D's front face (x 0.292) and its back face is 1.5 mm off the right post at the post's right side and
# 2.5 mm off at the post's centre. Overlap y -0.025 .. +0.020 (45 mm, middle at -2.5 mm); lap gap 2.3 .. 6.0 mm.
U_TURN = 0.0824  # rad
U_YAW = math.pi / 2 + U_TURN  # right arm, fingers closing along x (turned with the panel)
U_X, U_Y = 0.2894, -0.1302
U_HOVER = (0.220, -0.130)  # waits here, 69 mm in front of its place, until the left hand has let go and lifted
U_ENTRY_X = 0.284  # 5.4 mm in front of the place: it moves in with its bottom clear of the table, then down
LEFT_PARK = (0.200, 0.300)
RIGHT_PARK = (0.200, -0.300)
VIA = {"left": (0.300, 0.260, -math.pi / 4), "right": (0.265, -0.265, math.pi / 4)}  # turning on the way


def _pick(arm: str) -> list[Waypoint]:
    x, y = L.PANEL_START[f"panel_{arm}"]
    return [
        Waypoint(2.5, (x, y, CARRY_Z), OPEN, "over the panel", 0.0),
        Waypoint(2.0, (x, y, MID_Z), OPEN, "down toward it", 0.0),
        Waypoint(1.5, (x, y, PICK_Z), OPEN, "slowly down round its top", 0.0),
        Waypoint(0.6, (x, y, PICK_Z), OPEN, "settle", 0.0),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, "grip it across its thickness", 0.0),
        Waypoint(0.5, (x, y, PICK_Z), GRIP, "settle the grip", 0.0),
        Waypoint(2.0, (x, y, CARRY_Z), GRIP, "lift it", 0.0),
    ]


def _left() -> list[Waypoint]:
    """Left arm: the downstream panel D, set first, against the left and middle posts."""
    vx, vy, vyaw = VIA["left"]
    yaw = -math.pi / 2
    return _pick("left") + [
        Waypoint(1.75, (vx, vy, CARRY_Z), GRIP, "carry, turning it", vyaw),
        Waypoint(1.75, (D_ENTRY_X, D_Y, CARRY_Z), GRIP, "over its place, length along y", yaw),
        Waypoint(2.0, (D_ENTRY_X, D_Y, ENTRY_Z), GRIP, "down in front of the posts", yaw),
        Waypoint(2.0, (D_PLACE_X, D_Y, SLIDE_Z), GRIP, "slowly back against the left and middle posts", yaw),
        Waypoint(1.5, (D_PLACE_X, D_Y, PLACE_Z), GRIP, "slowly straight down", yaw),
        Waypoint(0.6, (D_PLACE_X, D_Y, PLACE_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (D_PLACE_X, D_Y, PLACE_Z), RELEASE, "let go", yaw),
        Waypoint(1.5, (D_PLACE_X, D_Y, LIFT_Z), RELEASE, "straight up", yaw),
        Waypoint(2.0, (*LEFT_PARK, PARK_Z), RELEASE, "park", yaw),
    ]


def _right() -> list[Waypoint]:
    """Right arm: the upstream panel U, set second, in front of D's end and against the right post."""
    vx, vy, vyaw = VIA["right"]
    left = _left()
    t_left_up = sum(w.duration for w in left[:-1])  # 23.2 s: the left hand has let go and lifted
    plan = _pick("right") + [
        Waypoint(1.75, (vx, vy, CARRY_Z), GRIP, "carry, turning it", vyaw),
        Waypoint(1.75, (*U_HOVER, CARRY_Z + 0.010), GRIP, "wait in front of its place", U_YAW),
    ]
    plan.append(Waypoint(t_left_up - sum(w.duration for w in plan), (*U_HOVER, CARRY_Z + 0.010), GRIP,
                         "wait until the left hand is up", U_YAW))
    return plan + [
        Waypoint(2.0, (U_ENTRY_X, U_HOVER[1], ENTRY_Z), GRIP, "down in front of D's end", U_YAW),
        Waypoint(1.5, (U_X, U_Y, SLIDE_Z), GRIP, "slowly in front of D's end, against the right post", U_YAW),
        Waypoint(1.5, (U_X, U_Y, PLACE_Z), GRIP, "slowly straight down", U_YAW),
        Waypoint(0.6, (U_X, U_Y, PLACE_Z), GRIP, "settle", U_YAW),
        Waypoint(1.5, (U_X, U_Y, PLACE_Z), RELEASE, "let go", U_YAW),
        Waypoint(1.5, (U_X, U_Y, LIFT_Z), RELEASE, "straight up", U_YAW),
        Waypoint(2.0, (*RIGHT_PARK, PARK_Z), RELEASE, "park", math.pi / 2),
    ]


def demo():
    """Both arms on one clock: the left arm sets D, then the right arm sets U in front of D's end.
    DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that
    arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
