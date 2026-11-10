"""On the station table: a fixed grey optical bench (60 mm wide, 6 mm tall, 560 mm long) runs along y, 30 cm in front of
the arm bases, with a black lamp box on its left (+y) end showing a black upward arrow (30 mm tall) on a white slide,
and black scale ticks on the table beside it every 50 mm from the arrow. On the bench stand two loose elements, each
on a rider: a converging lens (f = 100 mm) in a grey holder, 50 mm from the arrow, and a white screen, its face 280 mm
from the arrow. Rule: set them so that a sharp image of the arrow, the same size as the arrow, falls on the screen
(lens 2f = 200 mm from the arrow, screen 2f beyond the lens), both standing upright, square across the bench and on
its centre line. Sharp at another size 0.5; one element at its same-size mark 0.25; either tipped or off the bench 0.

The lens and the screen are loose mechkit compounds (mechkit.py, next to this file): each one rigid body of boxes, so
it stands on its rider and the scorer can read its pose (base frame and orientation).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the lens and screen are built as compound bodies)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s: an element counts once its rider has rested on the bench for 0.5 s, which
# can happen while the slow descent is still finishing (the arm sags a few mm under it) and with the gripper still
# closed. Settling and opening take 0.5 + 1.5 s, so the trial ends with the fingers open or nearly so.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Set up the converging lens (focal length 100 mm) and the white screen on the optical bench so that a sharp "
    "image of the lamp's arrow, the same size as the arrow, falls on the screen. Leave both standing upright on the "
    "bench, square across it and on its centre line, the screen's white face toward the lamp."
)


def _box(name: str, x: float, y: float, z0: float, size, color) -> Box:
    """A fixed box by its centre (x, y), its bottom height above the table and its size."""
    return Box(name, pos=(x, y, L.TOP + z0 + size[2] / 2), size=size, color=color)


BENCH = _box("optical_bench", L.BENCH_X, (L.BENCH_Y0 + L.BENCH_Y1) / 2, 0.0,
             (L.BENCH_W, L.BENCH_Y1 - L.BENCH_Y0, L.BENCH_H), L.ALUMINIUM_GREY)
LAMP = [
    _box("lamp_box", L.BENCH_X, L.LAMP_Y, L.BENCH_H, L.LAMP_SIZE, L.LAMP_BLACK),
    _box("arrow_slide", L.BENCH_X, L.SLIDE_Y, L.BENCH_H + L.AXIS_Z - L.SLIDE_SIZE[2] / 2, L.SLIDE_SIZE,
         L.SLIDE_WHITE),
]
LAMP += [_box(name, L.BENCH_X, L.OBJ_Y - L.ARROW_T / 2, L.BENCH_H + z0, (w, L.ARROW_T, z1 - z0), L.ARROW_BLACK)
         for name, w, z0, z1 in L.ARROW_PARTS]
TICKS = []
for _k in range(L.TICK_COUNT):
    _len = L.TICK_LONG if _k % 2 == 0 else L.TICK_SHORT
    TICKS.append(_box(f"scale_tick_{_k * 50}", L.TICK_X1 - _len / 2, L.OBJ_Y - _k * L.TICK_STEP, 0.0,
                      (_len, *L.TICK_SIZE_YZ), L.TICK_BLACK))


def _lens():
    """The lens in its holder on a rider, base frame at the centre of the rider's underside, square across the
    bench."""
    colors = [L.RIDER_BLACK] + [L.HOLDER_GREY] * 4 + [L.GLASS_BLUE]
    shapes = [mk.box(c, s, color=col) for (c, s), col in zip(L.LENS_BOXES, colors)]
    sx, sy = L.START["lens"]
    return mk.assembly("lens", origin=(sx, sy, L.BENCH_TOP), yaw=L.START_YAW, mass=L.LENS_MASS,
                       color=L.HOLDER_GREY, shapes=shapes)


def _screen():
    """The screen on a rider, base frame at the centre of the rider's underside, its white face toward the lamp."""
    colors = [L.RIDER_BLACK, L.SCREEN_WHITE, L.BACK_BLACK]
    shapes = [mk.box(c, s, color=col) for (c, s), col in zip(L.SCREEN_BOXES, colors)]
    sx, sy = L.START["screen"]
    return mk.assembly("screen", origin=(sx, sy, L.BENCH_TOP), yaw=L.START_YAW, mass=L.SCREEN_MASS,
                       color=L.SCREEN_WHITE, shapes=shapes)


LENS = _lens()
SCREEN = _screen()
OBJECTS = [BENCH, *LAMP, *TICKS, LENS, SCREEN]

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
    """A sharp, same-size image: lens and screen each standing on the bench (upright within 5 degrees, square
    within 5, centre within 8 mm of the centre line, at rest), the screen within 10 mm of v* = f u / (u - f), v / u
    from 0.9 to 1.1, neither tipped nor off the bench; held 2.5 s. This runs first every control step, so it also feeds
    the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    elements = {n: objects[n] for n in L.ELEMENTS}
    result = _TRACKER.update(elements)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, g in elements.items():
        try:
            row[n + "_state"] = {"source": getattr(g, "source", None), "origin": getattr(g, "origin", None),
                                 "quat": g.quat_xyzw, "vel": g.lin_vel}
        except AttributeError:
            row[n + "_state"] = "unreadable"
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the lens or the screen is tipped (over 10 degrees) or off the bench. Otherwise 1.0 for a sharp image the
    same size as the arrow, 0.5 for a sharp image of another size, 0.25 for one standing element at its same-size mark
    (lens 190-210 mm from the arrow, or screen 390-410 mm), else 0. Read from the state at hand (the last step)."""
    return _TRACKER.result({n: objects[n] for n in L.ELEMENTS}).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part. Both
# elements are 22 mm thick along the bench and are held with the fingers closing along y (yaw 0), across that
# thickness: the lens by its holder's top bar, above the glass, the screen by the top of its board.
LENS_OPEN = -0.025  # 50 mm: 14 mm either side of the holder; the pads stay 17 mm clear of the lamp's slide
SCREEN_OPEN = -0.030  # 60 mm
LENS_GRIP = -(L.HOLDER_T - 0.008) / 2  # 14 mm commanded across the 22 mm holder
SCREEN_GRIP = -(L.BOARD_T - 0.008) / 2  # 14 mm across the 22 mm board
YAW = 0.0
# The tool point's height above the rider's underside when gripping: the pad tips (13 mm below it) 8 mm above the
# lens's glass (64 mm up), 20 mm of pad on the holder's top bar; on the screen 27 mm of pad on the board.
GRIP_UP = {"lens": 0.085, "screen": 0.098}
LIFT = 0.020  # carried with the rider 20 mm above the bench
DROP = 0.002  # let go with the rider 2 mm above the bench
APPROACH = 0.030  # each final descent starts this far above the grip height
CARRY_Z = L.TOP + 0.150  # over the elements (the screen's top is 118 mm above the table)
# After letting go, straight up until the pad tips are this far above the element's top: joint 4 of the left arm loses
# its margin quickly over the lens's place (19.6 deg at the grip, 12.8 deg 8 mm over the lens, 4 deg at 15 cm up), so
# the left gripper then slides off the holder's end along -x (the open pads pass beside the holder, 13.5 mm clear of
# the glass) before it turns for home. The right gripper rises 20 mm over the screen and goes home at the carry height:
# going home at 12 cm, its lamp-side pad crossed the screen's top 2 mm too low (finger pads vs the boxes, offline).
RISE = {"lens": 0.008, "screen": 0.020}
LENS_BACK_OFF = 0.060  # the left gripper slides this far along -x, off the lens holder's end (x 0.275), before going home
PARK = {"left": (0.200, 0.300, L.TOP + 0.120), "right": (0.200, -0.300, CARRY_Z)}
# Trim (x, y), for the GPU run: if an element lands off its aim (centre_mm in TASK_TRACE away from the aim, or
# from_object_mm away from 200 / 400), move its place point by minus the offset.
PLACE_TRIM = {"lens": (0.0, 0.0), "screen": (0.0, 0.0)}
# The left arm waits on its own side for 8 s, so that it carries the lens toward the middle of the bench only once the
# right arm has let go of the screen. Working at once, the two grippers stay 21 cm apart, but the wrists (link 4)
# come within 3.7 cm of each other (capsule proxy, offline); with the wait they stay at least 15.8 cm apart while
# both work, and 14.0 cm at the end, the left arm going home past the parked right arm.
LEFT_WAIT = 8.0
ARM_OF = {"left": "lens", "right": "screen"}


def _arm(side: str, wait: float = 0.0) -> list[Waypoint]:
    """One arm: over its element, grip it, lift the rider 20 mm off the bench, carry it straight along the bench to
    its place, lower it slowly, let go, go up (the left gripper then slides off the holder's end) and back."""
    name = ARM_OF[side]
    opened, grip = (LENS_OPEN, LENS_GRIP) if name == "lens" else (SCREEN_OPEN, SCREEN_GRIP)
    px, py = L.START[name]
    ax, ay = L.AIM[name]
    qx, qy = ax + PLACE_TRIM[name][0], ay + PLACE_TRIM[name][1]
    pz = L.BENCH_TOP + GRIP_UP[name]
    lz = pz + LIFT
    qz = pz + DROP
    rz = L.BENCH_TOP + L.HEIGHTS[name] + 0.013 + RISE[name]  # the pad tips RISE above the element's top
    plan = []
    if wait > 0:
        plan.append(Waypoint(wait, PARK[side], opened, "wait on its own side while the other arm works", YAW))
    plan += [
        Waypoint(2.0, (px, py, CARRY_Z), opened, f"over the {name}", YAW),
        Waypoint(1.0, (px, py, pz + APPROACH), opened, f"down to just above the {name}", YAW),
        Waypoint(1.5, (px, py, pz), opened, f"slowly down round the {name}", YAW),
        Waypoint(0.5, (px, py, pz), opened, "settle", YAW),
        Waypoint(1.5, (px, py, pz), grip, f"grip the {name}", YAW),
        Waypoint(0.3, (px, py, pz), grip, "settle the grip", YAW),
        Waypoint(1.5, (px, py, lz), grip, "lift the rider off the bench", YAW),
        Waypoint(2.5, (qx, qy, lz), grip, "carry it along the bench to its place", YAW),
        Waypoint(0.5, (qx, qy, lz), grip, "settle over its place", YAW),
        Waypoint(2.0, (qx, qy, qz), grip, "slowly lower it onto the bench", YAW),
        Waypoint(0.5, (qx, qy, qz), grip, "settle", YAW),
        Waypoint(1.5, (qx, qy, qz), opened, f"let go of the {name}", YAW),
        Waypoint(0.5, (qx, qy, qz), opened, "settle", YAW),
        Waypoint(1.0, (qx, qy, rz), opened, "straight up", YAW),
    ]
    if name == "lens":
        plan += [
            Waypoint(0.8, (qx - LENS_BACK_OFF, qy, rz), opened, "slide off the holder's end", YAW),
            Waypoint(1.2, PARK[side], opened, "back toward its own side", YAW),
        ]
    else:
        plan.append(Waypoint(1.5, PARK[side], opened, "back toward its own side", YAW))
    return plan


def demo():
    """Both arms on one clock: the left sets the lens 200 mm from the arrow, the right the screen's face 400 mm from
    it. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that
    arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left", LEFT_WAIT), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
