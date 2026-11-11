"""On the station table, a scaled pipe-rack fit-up: two black steel riser stubs (vertical pipe ends, fixed), 36 mm
across on the left (+y) and 50 mm across on the right (-y), both 60 mm tall, 30 cm apart. Around each the plant's two
principal centrelines are painted in white on the floor, the north-south line along x (plant north +x, a white
square at its north end) and the east-west line along y, crossing at the stub's axis. Nearer the arms lie two loose
aluminium weld-neck flanges, neck down, painted steel grey with their bolt holes and bore painted dark on the face:
a 4-hole flange (60 mm disc, 36 mm neck) on the left and an 8-hole flange (70 mm disc, 50 mm neck) on the right,
both with a hole on each centreline's direction. Rule (pipefitters' 'two-holing'): each flange goes on the stub it is
bored for, neck down on the stub end, centred, with its bolt holes straddling the plant centrelines, never on them:
for 4 holes the pattern sits 45 degrees off the lines, for 8 holes 22.5 degrees.

Each flange is a loose mechkit compound (mechkit.py, next to this file): one rigid body of two cylinders (neck and
disc) and the thin painted discs on its face, so its hole pattern is visible and the scorer can read its pose (base
frame and orientation). The scorer also reads the arms (pad-tip points and openings) through rigstate.py, which wraps
the embodiment's step and reset from this file's import, only to check that each flange was let go.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the flanges are built as compound bodies)
import rigstate  # noqa: E402
from scoring import FLANGES, Tracker, hole_point  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0) must hold 1 s. Every credit already needs the flange let go (no pad-tip point lower than 10 mm above
# it within a finger's reach) and still for the last 1 s, so the trial cannot end while a gripper holds a flange.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Fit-up on the two riser stubs: set each weld-neck flange on the stub it is bored for, neck down on the stub end "
    "and centred, and two-hole it to the plant centrelines painted on the floor around the stub. The small black "
    "circles on each flange's face are its bolt holes. Then let go of both flanges and move the grippers clear."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


STUBS = [Cylinder(name, pos=(x, y, L.TOP + L.STUB_H / 2), radius=r, height=L.STUB_H, color=L.STUB_BLACK)
         for name, (x, y, r) in L.STUBS.items()]


def _centrelines(stub: str) -> list[Box]:
    """The painted cross around one stub: four white strips from the stub's surface out to 80 mm from its axis (N
    and S along x, E and W along y; west is +y, plant north +x) and a white square just past the north end."""
    x, y, r = L.STUBS[stub]
    w, t, e = L.LINE_W / 2, L.LINE_T, L.LINE_REACH
    tag = stub.split("_")[1]
    nx, ny = L.NORTH_MARK
    return [
        _span_box(f"line_n_{tag}", x + r, x + e, y - w, y + w, 0.0, t, L.LINE_WHITE),
        _span_box(f"line_s_{tag}", x - e, x - r, y - w, y + w, 0.0, t, L.LINE_WHITE),
        _span_box(f"line_w_{tag}", x - w, x + w, y + r, y + e, 0.0, t, L.LINE_WHITE),
        _span_box(f"line_e_{tag}", x - w, x + w, y - e, y - r, 0.0, t, L.LINE_WHITE),
        _span_box(f"north_mark_{tag}", x + e + 0.002, x + e + 0.002 + nx, y - ny / 2, y + ny / 2, 0.0, t,
                  L.LINE_WHITE),
    ]


LINES = [b for s in L.STUBS for b in _centrelines(s)]


def _flange(name: str):
    f = L.FLANGES[name]
    x, y = f["start"]
    shapes = [
        mk.cylinder((0.0, 0.0, L.NECK_H / 2), f["neck_r"], L.NECK_H),
        mk.cylinder((0.0, 0.0, L.NECK_H + L.DISC_T / 2), f["disc_r"], L.DISC_T),
        mk.cylinder((0.0, 0.0, L.FACE_Z + L.PAINT_T / 2), f["bore_r"], L.PAINT_T, color=L.BORE_DARK),
    ]
    for k in range(f["holes"]):
        a = 2 * math.pi * k / f["holes"]
        shapes.append(mk.cylinder((f["bc_r"] * math.cos(a), f["bc_r"] * math.sin(a), L.FACE_Z + L.PAINT_T / 2),
                                  f["hole_r"], L.PAINT_T, color=L.HOLE_BLACK))
    return mk.assembly(name, origin=(x, y, L.TOP), yaw=f["yaw"], mass=f["mass"], color=L.FLANGE_GREY, shapes=shapes)


FLANGE_4 = _flange("flange_4")
FLANGE_8 = _flange("flange_8")
OBJECTS = [*STUBS, *LINES, FLANGE_4, FLANGE_8]

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


def _states(objects) -> dict:
    return {n: objects[n] for n in FLANGES}


def success(objects) -> bool:
    """Both flanges seated on their own stubs (centred within 6 mm, neck on the stub end within 3 mm, level within
    5 degrees, still for 1 s, let go) and both two-holed; held 1 s. This runs first every control step, so it also
    feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    states = _states(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(states, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    try:
        for n in FLANGES:
            row[f"{n}_origin"] = states[n].origin
            row[f"{n}_quat"] = states[n].quat_xyzw
            row[f"{n}_source"] = getattr(states[n], "source", None)
            row[f"{n}_hole1"] = hole_point(states[n], n)
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tips=arms["left"].tips, right_tips=arms["right"].tips,
                   openings=[arms["left"].opening, arms["right"].opening])
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 per flange seated on the stub it is bored for (centred within 6 mm, neck on the stub end within 3 mm,
    level within 5 degrees, still for 1 s, no gripper on it) and 0.25 more per seated flange whose holes straddle
    the centrelines (4-hole within 8 degrees of 45 off the lines, 8-hole within 7 degrees of 22.5 off). Read from the
    state at hand (the last step)."""
    return _TRACKER.result(_states(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN_F = -0.04695  # fully open, 94 mm: 17 mm either side of the 60 mm disc, 12 mm either side of the 70 mm one
GRIP_F = {n: -(2 * f["disc_r"] - 0.008) / 2 for n, f in L.FLANGES.items()}  # 52 and 62 mm across the discs
# Each flange is gripped across its disc's rim, the pads outside the disc (and, at the place, outside the stub: the
# gripping pads are 30 and 35 mm from the axis, the stubs' surfaces 18 and 25 mm). A round disc gives the same width
# at any yaw, so the grip yaw is free: it is the pattern's heading plus a constant, and turning the gripper turns the
# hole pattern by the same angle.
GRIP_UP = 0.020  # the TCP 20 mm above the neck's bottom: pad tips 7 mm up, 3 mm under the disc, outside the neck
DROP = 0.001  # each flange is let go with its neck's bottom aimed 1 mm above the stub end
APPROACH = 0.030  # each final descent starts this far above its end
CARRY = L.TOP + 0.12  # carried with the neck's bottom 100 mm up: 40 mm over the stub ends
UP = L.TOP + 0.16  # straight up after letting go: pad tips about 65 mm above the flange's face
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.STUB_END_W + DROP + GRIP_UP
# Two-holed: hole 1 (and so the pattern) half a pitch off the centrelines. The 4-hole pattern repeats every 90
# degrees, so the left arm turns it 45 degrees clockwise (-45: the wrist then stays within 22 degrees of its bearing
# to the stub); the 8-hole pattern goes 22.5 degrees counter-clockwise. In our GPU runs gripped parts ended turned
# counter-clockwise of the commanded yaw by +0.6 to +1.7 degrees, so both yaws are commanded 1 degree clockwise of
# the answer (YAW_TRIM). If straddle_off_deg in TASK_TRACE at rest is over 2 degrees, move YAW_TRIM by minus the
# signed error.
YAW_TRIM = math.radians(-1.0)
YAW_PICK = 0.0
YAW_PLACE = {"flange_4": math.radians(-45.0) + YAW_TRIM, "flange_8": math.radians(22.5) + YAW_TRIM}
# Parts carried in a gripper landed about 4 mm short of the commanded point in our GPU runs (the arm trails 3-7 mm
# under load), so each place point leads its stub's axis by LEAD along the carry direction (offline, with no lag, a
# flange lands 4 mm off its stub's axis: inside the 6 mm). Trims (x, y), for the GPU run: if centre_mm in TASK_TRACE
# is over 3 mm, read flange_*_origin against the stub's axis and move the trim by minus the offset.
LEAD = 0.004
TRIM = {"flange_4": (0.0, 0.0), "flange_8": (0.0, 0.0)}
ARM_OF = {"flange_4": "left", "flange_8": "right"}
PARK = {"left": (0.20, 0.25, L.TOP + 0.16), "right": (0.20, -0.25, L.TOP + 0.16)}


def place_point(name: str) -> tuple[float, float]:
    """The TCP's x, y over the flange's stub: the stub's axis plus LEAD along the carry direction, plus the trim."""
    px, py = L.FLANGES[name]["start"]
    sx, sy, _ = L.STUBS[L.FLANGES[name]["stub"]]
    d = math.hypot(sx - px, sy - py)
    tx, ty = TRIM[name]
    return sx + LEAD * (sx - px) / d + tx, sy + LEAD * (sy - py) / d + ty


def _plan(name: str) -> list[Waypoint]:
    """Grip the flange across its rim, lift it, carry it over its stub turning the hole pattern half a pitch, lower
    it slowly onto the stub end, let go, go straight up and aside (19.0 s)."""
    px, py = L.FLANGES[name]["start"]
    qx, qy = place_point(name)
    grip, yaw = GRIP_F[name], YAW_PLACE[name]
    mid = ((px + qx) / 2, (py + qy) / 2, CARRY)
    return [
        Waypoint(2.0, (px, py, CARRY), OPEN_F, "over the flange", YAW_PICK),
        Waypoint(1.0, (px, py, PICK_Z + APPROACH), OPEN_F, "down to just above it", YAW_PICK),
        Waypoint(1.5, (px, py, PICK_Z), OPEN_F, "slowly down around its rim", YAW_PICK),
        Waypoint(0.5, (px, py, PICK_Z), OPEN_F, "settle", YAW_PICK),
        Waypoint(1.5, (px, py, PICK_Z), grip, "grip the flange", YAW_PICK),
        Waypoint(1.5, (px, py, CARRY), grip, "lift it", YAW_PICK),
        Waypoint(1.25, mid, grip, "carry it toward its stub, turning it", (YAW_PICK + yaw) / 2),
        Waypoint(1.25, (qx, qy, CARRY), grip, "over its stub, holes off the centrelines", yaw),
        Waypoint(0.8, (qx, qy, CARRY), grip, "settle over the stub", yaw),
        Waypoint(1.0, (qx, qy, PLACE_Z + APPROACH), grip, "down to 3 cm above the stub end", yaw),
        Waypoint(2.0, (qx, qy, PLACE_Z), grip, "slowly onto the stub end", yaw),
        Waypoint(0.5, (qx, qy, PLACE_Z), grip, "settle", yaw),
        Waypoint(1.5, (qx, qy, PLACE_Z), OPEN_F, "let go of it", yaw),
        Waypoint(1.2, (qx, qy, UP), OPEN_F, "straight up", yaw),
        Waypoint(1.5, PARK[ARM_OF[name]], OPEN_F, "aside", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: the 4-hole flange onto the 36 mm stub, its pattern turned 45 degrees clockwise."""
    return _plan("flange_4")


def _right() -> list[Waypoint]:
    """Right arm: the 8-hole flange onto the 50 mm stub, its pattern turned 22.5 degrees counter-clockwise."""
    return _plan("flange_8")


def demo():
    """Both arms at once. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
