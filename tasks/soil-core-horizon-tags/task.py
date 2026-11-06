"""On the station table: a flat matte black core tray (180 x 720 x 6 mm, fixed) runs along y, 32 to 50 cm in
front of the arm bases, and an opened soil core (a 50 mm wide, 30 mm tall band, 560 mm long, fixed) lies along
its middle from the ground surface at y +0.280 to 560 mm deep at y -0.280, in five coloured horizons (A, E, Bt,
BC, C) with boundaries at 46, 173, 314 and 434 mm deep. Four white depth tags (20 mm across, 40 mm tall, 15 g)
stand on the table, a pair at each end of the tray (each pair beyond its own arm's base). Task: stand one tag
upright on the tray beside the core at each boundary, let go and lift the grippers clear. Score 0.25 per boundary
tagged within 10 mm.

The scorer reads the arms (pad-tip points and openings) through rigstate.py, which wraps the embodiment's step and reset from
this file's import; it changes nothing the embodiment does. It is used only to check that each tag was let go.

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
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "This soil core lies open in the black tray. Stand one white tag upright on the tray right next to the core "
    "at each place where one soil layer meets the next, one tag per boundary. Then let go and lift the grippers "
    "clear."
)

TRAY = Box("core_tray", pos=(L.TRAY_X, L.TRAY_Y, L.TOP + L.TRAY_H / 2), size=L.TRAY_SIZE, color=L.TRAY_BLACK)
CORE = [
    Box(name, pos=(L.CORE_X, L.y_at((d0 + d1) / 2), L.TRAY_TOP + L.CORE_H / 2), size=(L.CORE_W, d1 - d0, L.CORE_H),
        color=L.HORIZON_RGB[horizon])
    for name, d0, d1, horizon in L.SEGMENTS
]
TAGS = [
    Cylinder(name, pos=(x, y, L.TOP + L.TAG_HEIGHT / 2), radius=L.TAG_RADIUS, height=L.TAG_HEIGHT, mass=L.TAG_MASS,
             color=L.TAG_WHITE)
    for name, x, y in L.TAG_STARTS
]
OBJECTS = [TRAY, *CORE, *TAGS]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the tags teleporting back)."""
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


def _tags(objects) -> dict:
    return {name: objects[name] for name in L.TAG_NAMES}


def success(objects) -> bool:
    """All four boundaries tagged: at each, a tag standing upright (within 10 deg) on the tray beside the core
    (centre outside it and at most 45 mm from its long edge), still (path under 3 mm over 0.5 s), let go and
    clear (no gripper's pad-tip point lower than 10 mm above its top with its axis within that gripper's finger
    reach: 75 mm fully open, 43 mm closed on a tag), centre within 10 mm of the boundary along the core. Held
    for 1 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate read for
    this step)."""
    arms = rigstate.arms()
    result = _TRACKER.update(_tags(objects), arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        row.update(left_tips=arms["left"].tips, right_tips=arms["right"].tips,
                   left_opening=arms["left"].opening, right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each of the four boundaries with a placed tag (upright, on the tray beside the core, still, let
    go) whose centre is within 10 mm of it along the core: 0, 0.25, 0.5, 0.75 or 1.0."""
    return _TRACKER.result(_tags(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
# Yaw 0: the fingers close along y, so they open along the core, away from it.
OPEN = -0.04695
TAG_GRIP = -(2 * L.TAG_RADIUS - 0.008) / 2  # 12 mm commanded on the 20 mm tag
TRAVEL_Z = L.TOP + 0.100  # not 0.12: the right arm's hover over B3, near the midline, keeps 0.22 rad of joint4
PICK_Z = L.TOP + 0.025  # TCP 25 mm up: pad tips 12 mm above the table, the pads on the tag's 12-34 mm
PLACE_Z = L.TRAY_TOP + 0.001 + 0.025  # the tag's bottom 1 mm above the tray when it is let go
PLACE_X = 0.360  # tag centre 25 mm from the core's near edge; the 27 mm wide pads stay 11 mm clear of it
# The 15 g tag hardly loads the arm (well under 1 mm of sag), so no place trim is needed. On the GPU read each
# tag's off_mm, x_mm and path_mm in TASK_TRACE.


def _move(tag_xy, place_xy, what: str) -> list[Waypoint]:
    """Pick one tag straight down, carry it at 10 cm, stand it on the tray beside the core, let go, go up: 15 s."""
    (tx, ty), (px, py) = tag_xy, place_xy
    return [
        Waypoint(2.5, (tx, ty, TRAVEL_Z), OPEN, f"over {what}"),
        Waypoint(1.5, (tx, ty, PICK_Z), OPEN, f"down round {what}"),
        Waypoint(0.5, (tx, ty, PICK_Z), OPEN, "settle"),
        Waypoint(1.5, (tx, ty, PICK_Z), TAG_GRIP, f"grip {what}"),
        Waypoint(1.5, (tx, ty, TRAVEL_Z), TAG_GRIP, "lift"),
        Waypoint(2.5, (px, py, TRAVEL_Z), TAG_GRIP, "carry it over its boundary"),
        Waypoint(2.0, (px, py, PLACE_Z), TAG_GRIP, "lower it slowly onto the tray beside the core"),
        Waypoint(0.5, (px, py, PLACE_Z), TAG_GRIP, "settle"),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go"),
        Waypoint(1.0, (px, py, TRAVEL_Z), OPEN, "straight up, clear of the tag"),
    ]


def _start(name: str) -> tuple[float, float]:
    return next((x, y) for n, x, y in L.TAG_STARTS if n == name)


def _place(k: int) -> tuple[float, float]:
    return PLACE_X, L.BOUNDARY_Y[k]


def _left() -> list[Waypoint]:
    """Left arm: its first tag to B1 (46 mm deep), then its second to B2 (173 mm); it then holds 10 cm above B2."""
    return _move(_start("tag_l1"), _place(0), "the first left tag") + _move(_start("tag_l2"), _place(1),
                                                                             "the second left tag")


def _right() -> list[Waypoint]:
    """Right arm, at the same time: its first tag to B3 (314 mm deep), then its second to B4 (434 mm); it then
    holds 10 cm above B4."""
    return _move(_start("tag_r1"), _place(2), "the first right tag") + _move(_start("tag_r2"), _place(3),
                                                                              "the second right tag")


def demo():
    """Both arms on one clock, each with its own two tags. The two grippers are 0.26 m or more apart in y whenever
    they are at the same x, so neither waits. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
