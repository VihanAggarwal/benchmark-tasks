"""On the station table (the floor of a nap room at 1:8): a green rest-area rug (240 x 480 mm, fixed) centred 28 cm in
front of the arm bases, and on it three blue nap cots (165 x 69 x 20 mm, loose) side by side, long side pointing
forward, with only 51 mm of floor between neighbours; on each cot a white pillow (30 x 44 x 12 mm, loose), all three at
the cots' far ends. Rule: spread the cots so neighbouring cots have at least 114 mm (3 ft) of floor between them,
each flat on the rug and still pointing forward, and set the pillows head to toe: one pillow on each cot, at one end of
it, no two neighbouring cots with their pillows at the same end.

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
from scoring import NAMES, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Held for 2.5 s, not the default 1 s. The last pillows count as on their cots while still gripped 2 mm above them
# (the scorer cannot see the grippers), 0.5 s after they stop; with 2.5 s the trial cannot end before the fingers have
# opened (0.3 + 1.5 s after the stop), so the end state is the pillows lying free.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Get the nap room ready. Spread the three cots so that there are at least 114 mm (3 feet at this 1:8 scale) of "
    "floor between neighbouring cots, keeping each one flat on the green rug and pointing the way it does now, and "
    "set the pillows head to toe: one pillow on each cot, at one end of it, and no two neighbouring cots with their "
    "pillows at the same end."
)

RUG = Box("rest_rug", pos=(L.RUG_X, L.RUG_Y, L.TOP + L.RUG_SIZE[2] / 2), size=L.RUG_SIZE, color=L.RUG_GREEN)
COTS = [
    Box(n, pos=(*L.COT_START[n], L.RUG_TOP + L.COT_SIZE[2] / 2), size=L.COT_SIZE, mass=L.COT_MASS,
        color=L.COT_BLUE)
    for n in L.COTS
]
PILLOWS = [
    Box(n, pos=(*L.PILLOW_START[n], L.COT_TOP + L.PILLOW_SIZE[2] / 2), size=L.PILLOW_SIZE, mass=L.PILLOW_MASS,
        color=L.PILLOW_WHITE)
    for n in L.PILLOWS
]
OBJECTS = [RUG, *COTS, *PILLOWS]

_TRACKER = Tracker()


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
    """Every cot set down on the rug (flat, pointing forward within 15 degrees, at rest), one pillow lying on each,
    both neighbouring pairs at least 114 mm apart with their pillows at opposite ends; held 2.5 s. This runs first
    every control step, so it also feeds the tracker."""
    states = {n: objects.get(n) for n in NAMES}
    result = _TRACKER.update(states)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    raw = {}
    for n, s in states.items():
        try:
            raw[n] = {"pos": s.pos, "quat": s.quat_xyzw, "vel": s.lin_vel}
        except AttributeError:
            raw[n] = "unreadable"
    row["raw"] = raw
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any cot is not set down (tipped, stacked, turned more than 15 degrees, off the rug, moving), any pillow is
    not lying on a cot, or two pillows are on one cot. Otherwise, for each of the two neighbouring pairs of cots,
    0.25 if at least 114 mm of floor separates them and 0.25 if their pillows mark opposite ends. Read from the state at
    hand (the last step)."""
    return _TRACKER.result({n: objects.get(n) for n in NAMES}).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Both arms at once, mirror images: each works only on its own side's cot. A pillow cannot be trusted to ride on a
# lifted cot, so each arm first lifts its cot's pillow off and sets it on the rug just past the cots' far ends, then
# carries the cot out toward the rug's edge, then lays the pillow on the cot's NEAR end. The middle cot and its
# pillow (far end) are not touched: near, far, near is head to toe.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part. Yaw 0: the
# fingers close along y, across the cot's 69 mm width and the pillow's 44 mm.
PILLOW_OPEN = -0.035  # 70 mm: 13 mm each side of the pillow
PILLOW_GRIP = -(L.PILLOW_SIZE[1] - 0.008) / 2
COT_OPEN = -0.04695  # fully open, 94 mm: 12 mm each side of the cot; at the start the finger on the middle cot's side
# is 32 mm clear of it
COT_GRIP = -(L.COT_SIZE[1] - 0.008) / 2
PAD = 0.013  # the pad tips are this far below the tool point
HOVER = L.TOP + 0.100  # travel height: a carried pillow's underside 83 mm up, 61 mm over the cots' tops
APPROACH = 0.030  # each final descent starts this far above its end point and takes 1.5 s
PILLOW_ON_COT_Z = L.COT_TOP + 0.004 + PAD  # pad tips 4 mm above the cot's top: 8 mm of the pillow's side
PILLOW_ON_RUG_Z = L.RUG_TOP + 0.004 + PAD
DROP = 0.002  # parts are let go this far above where they come to rest
COT_GRIP_Z = L.RUG_TOP + 0.007 + PAD  # pad tips 7 mm above the cot's underside: 13 mm of its side
COT_LIFT_Z = L.TOP + 0.065  # the carried cot's underside 43 mm above the rug
SIDE = {"left": 1.0, "right": -1.0}
COT_OUT_Y = 0.194  # where each outer cot is set: 125 mm from the middle cot, 11.5 mm inside the rug's edge
PARK = (0.383, 0.130)  # where a pillow waits on the rug (x 0.368-0.398), 8 mm past the cots' far ends (x 0.360)
NEAR_X = L.COT_START["cot_middle"][0] - L.COT_HALF_LEN + L.END_INSET  # a pillow 25 mm in from a cot's near end
REST = (0.250, 0.280, L.TOP + 0.120)  # where each arm waits at the end, clear of the cots
# Trim (x, y) of the cot and the pillow place points, for the GPU run: if a cot or a pillow lands off its aim
# (centre_mm in TASK_TRACE away from (277.5, +-194.0) for a cot or (220.0, +-194.0) for a pillow), move its place point
# by minus the offset, on the side it is on (mirror y for the right arm).
COT_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
PILLOW_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _pick_place(src, src_z, dst, dst_z, open_, grip, lift_z, what: str, carry_s: float = 1.5) -> list[Waypoint]:
    """Over the part, down (slowly for the last 30 mm), grip, lift, carry, down (slowly), let go, up."""
    return [
        Waypoint(1.5, (*src, HOVER), open_, f"over the {what}"),
        Waypoint(1.0, (*src, src_z + APPROACH), open_, f"down to just above the {what}"),
        Waypoint(1.5, (*src, src_z), open_, f"slowly down round the {what}"),
        Waypoint(0.5, (*src, src_z), open_, "settle"),
        Waypoint(1.5, (*src, src_z), grip, f"grip the {what}"),
        Waypoint(0.3, (*src, src_z), grip, "settle the grip"),
        Waypoint(1.2, (*src, lift_z), grip, f"lift the {what}"),
        Waypoint(carry_s, (*dst, lift_z), grip, f"carry the {what}"),
        Waypoint(0.5, (*dst, lift_z), grip, "settle over the place"),
        Waypoint(1.0, (*dst, dst_z + APPROACH), grip, "down to just above the place"),
        Waypoint(1.5, (*dst, dst_z), grip, f"slowly set the {what} down"),
        Waypoint(0.3, (*dst, dst_z), grip, "settle"),
        Waypoint(1.5, (*dst, dst_z), open_, f"let go of the {what}"),
        Waypoint(1.0, (*dst, HOVER), open_, "straight up"),
    ]


def _arm(side: str) -> list[Waypoint]:
    """One arm, on its own side: the pillow off its cot onto the rug past the far ends, the cot out toward the rug's
    edge, the pillow onto the cot's near end; then up and out of the way."""
    k = SIDE[side]
    cot = L.COT_START["cot_" + side]
    pillow = L.PILLOW_START["pillow_" + side]
    park = (PARK[0], k * PARK[1])
    cot_to = (cot[0] + COT_TRIM[side][0], k * COT_OUT_Y + COT_TRIM[side][1])
    pillow_to = (NEAR_X + PILLOW_TRIM[side][0], k * COT_OUT_Y + PILLOW_TRIM[side][1])
    plan = _pick_place(pillow, PILLOW_ON_COT_Z, park, PILLOW_ON_RUG_Z + DROP, PILLOW_OPEN, PILLOW_GRIP, HOVER,
                       "pillow")
    plan += _pick_place(cot, COT_GRIP_Z, cot_to, COT_GRIP_Z + DROP, COT_OPEN, COT_GRIP, COT_LIFT_Z, "cot", 2.0)
    plan += _pick_place(park, PILLOW_ON_RUG_Z, pillow_to, PILLOW_ON_COT_Z + DROP, PILLOW_OPEN, PILLOW_GRIP, HOVER,
                        "pillow", 2.0)
    plan.append(Waypoint(1.5, (REST[0], k * REST[1], REST[2]), PILLOW_OPEN, "out of the way"))
    return plan


def demo():
    """Both arms on one clock, mirror images, each on its own side of the midline: the two tool points never come
    closer than 0.24 m (the outer cots start at y +-0.12 and go out to +-0.194). DEMO_HOLD_ONLY=1: no plan (both arms
    hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm
    check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
