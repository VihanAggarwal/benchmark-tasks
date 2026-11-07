"""On the station table: a fixed oak shoe rail (356 mm long along y, 40 mm wide, 10 mm tall) centred 30 cm in front of
the arm bases on the midline, running face to face between two fixed oak newel posts (40 x 40 mm, 100 mm tall), and
six loose white balusters (upright cylinders 20 mm across, 80 mm tall, 18 g) standing on the table, three on each
arm's side. The balusters go upright on the rail so that no opening, the two next to the posts included, passes a
51 mm test ball (the 4 in sphere rule for residential guards, at 1:2), spaced evenly. Four can never do it (five
openings of 55.2 mm even when perfectly even), five (42.7 mm) or six (33.7 mm) can. Gripping from above, neither
arm reaches the other side's balusters or the far end of the rail.

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
from scoring import Tracker, bottom_gap, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. The last baluster counts as standing (bottom within 2 mm of the rail) while the
# gripper still holds it, from the end of its slow descent; settling and letting go take 0.5 + 1.5 s more. With 3 s
# the trial cannot end while the fingers are still closed on it.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Stand the white balusters upright on the oak shoe rail between the two newel posts so that a 51 mm test ball "
    "cannot pass through any gap, including the gaps next to the posts, and space them evenly. Use as many of the "
    "six balusters as you need."
)

RAIL = Box("shoe_rail", pos=(L.RAIL_X, 0.0, L.TOP + L.RAIL_H / 2), size=(L.RAIL_W, L.RAIL_LEN, L.RAIL_H),
           color=L.OAK)
POSTS = [
    Box(name, pos=(L.RAIL_X, y, L.TOP + L.POST_SIZE[2] / 2), size=L.POST_SIZE, color=L.OAK)
    for name, y in L.POSTS
]
BALUSTERS = [
    Cylinder(name, pos=(x, y, L.TOP + L.BAL_H / 2), radius=L.BAL_R, height=L.BAL_H, mass=L.BAL_MASS, color=L.WHITE)
    for name, x, y in L.BALUSTERS
]
OBJECTS = [RAIL, *POSTS, *BALUSTERS]
BAL_NAMES = tuple(b.name for b in BALUSTERS)

_TRACKER = Tracker()


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


def _pose(s) -> dict:
    """A baluster's centre (mm), tilt (deg), bottom above the rail's top (mm) and speed (mm/s) for the trace."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "bottom_mm": round(bottom_gap(s) * 1000, 1),
                "speed_mm_s": round(float(sum(float(v) ** 2 for v in s.lin_vel) ** 0.5) * 1000, 1)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """No opening along the rail (post to baluster, baluster to baluster) of 51 mm or more, widest minus narrowest
    opening at most 10 mm, no baluster down on the guard, every baluster still (SUCCESS_HOLD_S = 3 s). This runs
    first every control step, so it also feeds the tracker."""
    bals = [objects[n] for n in BAL_NAMES]
    result = _TRACKER.update(bals)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    row.update({n: _pose(objects[n]) for n in BAL_NAMES})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a baluster lies on or leans against the rail or a post without standing. Else 1.0 no opening passes
    the 51 mm ball and the openings are within 10 mm of each other; 0.5 no opening passes; 0.25 exactly one opening
    passes with four or more balusters standing; 0 otherwise. Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in BAL_NAMES]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.020  # 40 mm: 10 mm either side of a baluster, and the pads stay 22 mm clear of its neighbours at the
#                start, which stand 60 mm away along the fingers' closing axis (fully open, 94 mm, would hit them)
GRIP = -(L.BAL_D - 0.008) / 2  # 12 mm commanded on the 20 mm baluster: about 20 N per finger
# Fingers close along x (across the rail), so the pads never come near a neighbour on the rail (62.7 mm away along
# y): left arm yaw -pi/2, right arm +pi/2 (the other signs run joint6 into its limit near the midline).
YAW = {"left": -math.pi / 2, "right": math.pi / 2}
GRIP_UP = L.BAL_H / 2  # the TCP at the baluster's mid-height: the pads hold 27-49 mm up its 80 mm
DROP = 0.001  # each baluster is let go with its bottom 1 mm above the rail's top
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.RAIL_TOP + DROP + GRIP_UP
ENTRY_Z = L.TOP + 0.100  # over a baluster on the table, the pad tips 7 mm above its top; the rest is slow
HOVER_Z = L.TOP + 0.140  # carried with its bottom 100 mm up: 20 mm over the standing balusters' tops
LOW_Z = L.TOP + 0.070  # moved forward over the rail with its bottom 30 mm up: 20 mm over the rail's top
UP_Z = L.TOP + 0.120  # after letting go: the pad tips 17 mm above the standing baluster's top
# The carried baluster reaches its line along y behind the posts (their back faces at x 0.28), so it never passes
# near a post, then comes forward along that line, low, between its neighbours (42.7 mm clear on each side).
LANE_X = 0.200
# Where each baluster goes (its centre's y on the rail's centre line), and in which order. The left arm sets three,
# the right arm two; the right arm's third baluster stays on the table (it is not needed).
PLACE_TO = {"baluster_l1": L.DEMO_YS[0], "baluster_l2": L.DEMO_YS[1], "baluster_l3": L.DEMO_YS[2],
            "baluster_r1": L.DEMO_YS[4], "baluster_r2": L.DEMO_YS[3]}
ORDER = {"left": ("baluster_l1", "baluster_l2", "baluster_l3"), "right": ("baluster_r1", "baluster_r2")}
# Per-arm trim of the place point (x, y), for the GPU run: if one arm's balusters land off their marks by the same
# amount (the pads off the commanded point), read the balusters' pos_mm in TASK_TRACE and move that arm's place
# point by minus the offset. An 18 g baluster held at its middle makes the arm sag well under 1 mm. Offline there is
# nothing to trim.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _wait(wps: list[Waypoint], until: float, label: str) -> None:
    """Hold the last pose until ``until`` s on the shared clock (no-op if already past it)."""
    t = _span(wps)
    if until > t + 1e-9:
        w = wps[-1]
        wps.append(Waypoint(until - t, w.pos, w.finger, label, w.yaw))


def _pick(name: str, arm: str, move: float) -> list[Waypoint]:
    """Over a baluster on the table, down round it, grip it at mid-height, lift it straight up."""
    (x, y), yaw = L.START[name], YAW[arm]
    return [
        Waypoint(move, (x, y, HOVER_Z), OPEN, f"over {name}", yaw),
        Waypoint(1.0, (x, y, ENTRY_Z), OPEN, "down to just above it", yaw),
        Waypoint(1.8, (x, y, PICK_Z), OPEN, "slowly down round it", yaw),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, "grip the baluster at mid-height", yaw),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", yaw),
    ]


def _place(name: str, arm: str) -> list[Waypoint]:
    """Carry it behind the posts to its line, down, forward over the rail, stand it on the rail, let go, back."""
    tx, ty = PLACE_TRIM[arm]
    x, y, yaw = L.RAIL_X + tx, PLACE_TO[name] + ty, YAW[arm]
    return [
        Waypoint(2.0, (LANE_X, y, HOVER_Z), GRIP, "carry it behind the posts to its line", yaw),
        Waypoint(1.2, (LANE_X, y, LOW_Z), GRIP, "down to 2 cm over the rail's height", yaw),
        Waypoint(1.5, (x, y, LOW_Z), GRIP, "forward over the rail", yaw),
        Waypoint(1.5, (x, y, PLACE_Z), GRIP, "slowly stand it on the rail", yaw),
        Waypoint(0.5, (x, y, PLACE_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (x, y, PLACE_Z), OPEN, "let go of it", yaw),
        Waypoint(0.3, (x, y, PLACE_Z), OPEN, "settle", yaw),
        Waypoint(1.0, (x, y, UP_Z), OPEN, "up, clear of its top", yaw),
        Waypoint(1.2, (LANE_X, y, HOVER_Z), OPEN, "back behind the posts", yaw),
    ]


def _plans() -> dict[str, list[Waypoint]]:
    """Both arms on one clock. Two grippers at the rail are never closer than 0.25 m along it:
      right: r1 -> y -0.1253, then r2 -> y -0.0627, then out of the way on its own side
      left:  l1 -> y +0.1253 (with r1); picks l2 while r2 goes in, carries it to its line once the right arm is back
             behind the posts -> y +0.0627; then l3 -> y 0"""
    right: list[Waypoint] = []
    for i, name in enumerate(ORDER["right"]):
        right += _pick(name, "right", 2.0 if i == 0 else 1.5) + _place(name, "right")
    t_right_back = _span(right)
    right.append(Waypoint(1.5, (0.160, -0.360, HOVER_Z), OPEN, "out of the way on its own side", YAW["right"]))

    l1, l2, l3 = ORDER["left"]
    left = _pick(l1, "left", 2.0) + _place(l1, "left") + _pick(l2, "left", 1.5)
    _wait(left, t_right_back, "hold it up until the right arm is back behind the posts")
    left += _place(l2, "left") + _pick(l3, "left", 1.5) + _place(l3, "left")
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = _plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
