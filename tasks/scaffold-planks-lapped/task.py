"""On the station table, a 1:8 model of two bays of a scaffold platform: three fixed galvanised putlogs (steel bars
120 mm along x, 20 x 20 mm in section) centred at x 26 cm and y +19, 0 and -19 cm, so the bays are 190 mm on
centre; and two loose orange scaffold boards (250 x 40 x 16 mm, 80 g) lying flat along x on the table at x 24.5 cm,
y +32 and -32 cm, one on each arm's side. Rule (29 CFR 1926.451(b)(4), (b)(5)(i), (b)(7) at 1:8): each bay planked
by a board, every board end 19 to 38 mm past the centre line of the last putlog under it, the boards in one straight
run lapped over the middle putlog. A board end more than 38 mm past its putlog (a trap end) scores 0.

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

import yam_arena  # noqa: E402

import layout as L  # noqa: E402
from scoring import Tracker, heading_err_deg, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold 1 s. The lapped board comes to rest on its putlog only once the gripper lets go of it (held
# level, its outer end is 16 mm above the putlog), and 'still' needs 0.5 s of rest, so the trial cannot end before
# the upper board has been let go.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Plank both scaffold bays with the two orange boards in one straight run: lay one board across each bay on the "
    "grey putlogs, lap the two boards over the middle putlog, and leave every board end 19 to 38 mm past the centre "
    "line of the last putlog under it."
)

PUTLOGS = [
    Box(name, pos=(L.PUTLOG_X, y, L.TOP + L.PUTLOG_SIZE[2] / 2), size=L.PUTLOG_SIZE, color=L.GALVANISED)
    for name, y in L.PUTLOGS
]
BOARDS = [
    Box(name, pos=(L.BOARD_START_X, y, L.TOP + L.BOARD_T / 2), size=L.BOARD_SIZE, mass=L.BOARD_MASS,
        color=L.BOARD_ORANGE)
    for name, y in L.BOARDS
]
OBJECTS = [*PUTLOGS, *BOARDS]
BOARD_NAMES = tuple(b.name for b in BOARDS)

# Offline only: a kinematic checker without physics keeps every object level, so the lapped board would hang level
# off the lower board's end; offline_settle gives it the pose it settles in (turned down onto its putlog). The real
# yam_arena has no OFFLINE_STUB, so in the simulator this does nothing and the physics settles the board.
if getattr(yam_arena, "OFFLINE_STUB", False):
    import offline_settle  # noqa: E402

    offline_settle.attach(BOARDS)

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


def _boards(objects) -> dict:
    return {n: objects[n] for n in BOARD_NAMES}


def _pose(s) -> dict:
    """A board's centre (mm), tilt and heading error from the run (deg) and speed (mm/s) for the trace."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s), 2),
                "heading_err_deg": round(heading_err_deg(s), 2),
                "speed_mm_s": round(float(sum(float(v) ** 2 for v in s.lin_vel) ** 0.5) * 1000, 1)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """A board bears in each bay (on both its putlogs, both ends 19-38 mm past their centre lines, along the run,
    tilted 8 degrees at most, still), the two in line within 10 mm and lapped over the middle putlog; held 1 s.
    This runs first every control step, so it also feeds the tracker."""
    boards = _boards(objects)
    result = _TRACKER.update(boards)
    row = {"score": result.score, "success": result.success, **result.details}
    row.update({f"{n}_pose": _pose(objects[n]) for n in BOARD_NAMES})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a board on the scaffold has an end more than 38 mm past its putlog (a trap end). Else 1.0 both bays
    planked and lapped in line; 0.5 both bays planked, not lapped in line; 0.25 one bay planked; 0 otherwise. Read
    from the state at hand (the last step)."""
    return _TRACKER.result(_boards(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.035  # 70 mm: 15 mm either side of a 40 mm board
GRIP = -(L.BOARD_W - 0.008) / 2  # 32 mm commanded across the board's 40 mm width
# Each board is picked where it lies along x with the fingers closing along y (yaw 0) and turned a quarter turn on
# the way, to be set down along the run (y) with the fingers closing along x: the left arm at yaw -pi/2, the right
# arm at +pi/2 (the other signs run joint6 into its limit near the midline).
YAW_PICK, YAW_LEFT, YAW_RIGHT = 0.0, -math.pi / 2, math.pi / 2
PAD = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.004 + PAD  # the TCP over a board's underside: pad tips 4 mm above it, holding 12 of its 16 mm
DROP = 0.002  # each board is let go with its underside 2 mm above what it lands on
APPROACH = 0.030  # each final descent starts this far above and takes 1.5 s
HOVER = L.TOP + 0.10  # carried with the board's underside 83 mm up, 47 mm over the finished lap
PARK_Z = L.TOP + 0.16

PICK_Z = L.TOP + GRIP_UP
LOWER_Z = L.LOWER_BOTTOM + DROP + GRIP_UP  # the right board onto the right and middle putlogs
UPPER_Z = L.LOWER_TOP + DROP + GRIP_UP  # the left board, level, 2 mm over the right board's end; let go, it tips
#                                         5.4 degrees about that end until its outer end rests on the left putlog


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _pick(y0: float, move: float) -> list[Waypoint]:
    x0 = L.BOARD_START_X
    return [
        Waypoint(move, (x0, y0, HOVER), OPEN, "over the board", YAW_PICK),
        Waypoint(1.0, (x0, y0, PICK_Z + APPROACH), OPEN, "down to just above the board", YAW_PICK),
        Waypoint(1.5, (x0, y0, PICK_Z), OPEN, "slowly down beside the board", YAW_PICK),
        Waypoint(0.5, (x0, y0, PICK_Z), OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (x0, y0, PICK_Z), GRIP, "grip the board across its width", YAW_PICK),
        Waypoint(1.2, (x0, y0, HOVER), GRIP, "lift the board", YAW_PICK),
    ]


def _place(y: float, z: float, yaw: float, what: str) -> list[Waypoint]:
    x = L.RUN_X
    return [
        Waypoint(2.0, (x, y, HOVER), GRIP, f"carry it over {what}, turning it along the run", yaw),
        Waypoint(1.0, (x, y, z + APPROACH), GRIP, "down to 3 cm above", yaw),
        Waypoint(1.5, (x, y, z), GRIP, f"slowly set it down {what}", yaw),
        Waypoint(0.5, (x, y, z), GRIP, "settle", yaw),
        Waypoint(1.5, (x, y, z), OPEN, "let go", yaw),
        Waypoint(1.0, (x, y, HOVER), OPEN, "up", yaw),
    ]


def _wait(wps: list[Waypoint], until: float, label: str) -> None:
    """Hold the last pose until ``until`` s on the shared clock (no-op if already past it)."""
    t = _span(wps)
    if until > t + 1e-9:
        w = wps[-1]
        wps.append(Waypoint(until - t, w.pos, w.finger, label, w.yaw))


def _plans() -> dict[str, list[Waypoint]]:
    """Both arms on one clock. The right arm lays its board across the right bay (right and middle putlogs, centred
    at y -0.095, ends 30 mm past both centre lines) and parks; meanwhile the left arm has picked its board and waits
    over its start; once the right arm is on its way out it laps its board over the right one's end (centred at
    y +0.095) and lets go."""
    right = _pick(L.BOARD_START_Y["board_right"], 2.0)
    right += _place(L.LOWER_Y, LOWER_Z, YAW_RIGHT, "across the right bay")
    t_out = _span(right)
    right.append(Waypoint(1.5, (0.25, -0.30, PARK_Z), OPEN, "park on its own side", YAW_RIGHT))

    left = _pick(L.BOARD_START_Y["board_left"], 2.0)
    _wait(left, t_out + 0.5, "wait over the start for the right arm to leave")
    left += _place(L.UPPER_Y, UPPER_Z, YAW_LEFT, "across the left bay, lapped on the right board's end")
    left.append(Waypoint(1.5, (0.25, 0.30, PARK_Z), OPEN, "park on its own side", YAW_LEFT))
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = _plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
