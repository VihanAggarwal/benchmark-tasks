"""On the station table: two fixed timber bearers (20 mm wide along x, 320 mm long along y, 20 mm tall) on the
midline at x 20 and 36 cm; four loose red-cedar boards (240 mm along x, 40 mm wide, 16 mm thick, 60 g), two on each
arm's side at y +-38 and +-46 cm, centred at x 28 cm; and two loose dark stickers (thin strips 20 mm wide, 280 mm
long, 16 mm thick, 40 g) lying along x outboard of each arm's boards at y +-54 cm. The boards go across the bearers
in two layers of two with an air gap, and a sticker between the layers right above each bearer, as lumber is
stacked to air-dry. Neither arm can reach the other side's sticker.

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
from scoring import BOARD_HALF, STICKER_HALF, Tracker, heading_err_deg, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. The last board rests on the stickers (within 3 mm) while the gripper still holds it,
# from shortly before the slow descent ends; settling, letting go and lifting the hand take 0.5 + 1.5 + 1.0 s more.
# With 3 s the trial cannot end while the fingers are still closed on the board.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Lay two of the pale boards across the two fixed bearers with a gap of about 6 cm between them, lay a dark "
    "sticker across those boards directly over each bearer, then lay the other two boards on the stickers, each "
    "right above one of the first two."
)

BEARERS = [
    Box(name, pos=(x, L.BEARER_Y, L.TOP + L.BEARER_SIZE[2] / 2), size=L.BEARER_SIZE, color=L.WEATHERED)
    for name, x in zip(("bearer_near", "bearer_far"), L.BEARER_XS, strict=True)
]
BOARDS = [
    Box(name, pos=(L.BOARD_START_X, y, L.TOP + L.BOARD_SIZE[2] / 2), size=L.BOARD_SIZE, mass=L.BOARD_MASS,
        color=L.CEDAR)
    for name, y in L.BOARDS
]
STICKERS = [  # sized as they lie in the stack (along y), turned a quarter turn to lie along x at the start
    Box(name, pos=(x, y, L.TOP + L.STICKER_SIZE[2] / 2), size=L.STICKER_SIZE, mass=L.STICKER_MASS,
        color=L.DARK_STAIN, yaw=L.STICKER_START_YAW)
    for name, x, y in L.STICKERS
]
OBJECTS = [*BEARERS, *BOARDS, *STICKERS]
BOARD_NAMES = tuple(b.name for b in BOARDS)
STICKER_NAMES = tuple(k.name for k in STICKERS)

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


def _pieces(objects):
    return [objects[n] for n in BOARD_NAMES], [objects[n] for n in STICKER_NAMES]


def _pose(s, half, axis: str) -> dict:
    """A piece's centre (mm), tilt and heading error (deg) and speed (mm/s) for the trace."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "heading_err_deg": round(heading_err_deg(s, half, axis), 2),
                "speed_mm_s": round(float(sum(float(v) ** 2 for v in s.lin_vel) ** 0.5) * 1000, 1)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """Two boards on the bearers with a 30-90 mm gap, a sticker on them within 10 mm of each bearer, two boards on
    the stickers within 15 mm of the lower ones, every piece still (SUCCESS_HOLD_S = 3 s). This runs first every
    control step, so it also feeds the tracker."""
    boards, stickers = _pieces(objects)
    result = _TRACKER.update(boards, stickers)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    row.update({n: _pose(objects[n], BOARD_HALF, "x") for n in BOARD_NAMES})
    row.update({n: _pose(objects[n], STICKER_HALF, "y") for n in STICKER_NAMES})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a board or sticker ends off the table or on an edge. Else 1.0 the full stack (layer 1 on both bearers
    with a 30-90 mm gap, stickers on it within 10 mm of different bearers, layer 2 on the stickers within 15 mm of
    layer 1); 0.75 the same with a sticker 10-25 mm off or a top board 15-30 mm off; 0.5 layer 1 and both stickers
    right, layer 2 not; 0.25 layer 1 only; 0 otherwise. Read from the state at hand (the last step)."""
    boards, stickers = _pieces(objects)
    return _TRACKER.result(boards, stickers).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
BOARD_OPEN = -0.035  # 70 mm: 15 mm either side of a 40 mm board, the pads clear of the neighbouring board
BOARD_GRIP = -(L.BOARD_SIZE[1] - 0.008) / 2  # yaw 0: the fingers close along y, across the board's 40 mm
STICKER_OPEN = -0.025  # 50 mm: 15 mm either side of a 20 mm sticker
STICKER_GRIP = -(L.STICKER_SIZE[0] - 0.008) / 2  # 12 mm commanded across the sticker's 20 mm width
# A sticker is picked where it lies along x with the fingers closing along y (yaw 0) and turned a quarter turn on
# the way, to be set down along y with the fingers closing along x: left arm yaw -pi/2, right arm +pi/2 (the other
# signs run joint6 into its limit near the midline).
YAW_BOARD, YAW_STICK_L, YAW_STICK_R = 0.0, -math.pi / 2, math.pi / 2
PAD = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.004 + PAD  # the TCP over a piece's underside: pad tips 4 mm above it, holding 12 of its 16 mm
DROP = 0.002  # each piece is let go with its underside 2 mm above its support (the arm sags a little under it)
APPROACH = 0.030  # each final descent starts this far above the grip height and takes 1.5 s
HOVER = L.TOP + 0.12  # carried with the piece's underside 103 mm up: 35 mm over the finished stack
# Each arm holds its sticker 10 cm from the sticker's centre, toward the arm bases (x - 0.10 where it lies along x).
# The quarter turn brings that end to the arm's own side (y +-0.10 when the sticker is centred on the midline): at
# the place the pads (27 mm wide) are 1.5 cm outboard of the layer-1 boards' edge (y +-0.07), over nothing, and the
# grip stays well inside reach. Off centre, the 40 g strip puts 0.04 N m on the grip, about a seventh of what the
# pads' friction holds (20 N per pad, friction 1.2, about 6 mm of lever over the patch).
STICK_OFF = 0.10

BOARD_PICK_Z = L.TOP + GRIP_UP
LAYER1_Z = L.LAYER1_BOTTOM + DROP + GRIP_UP
LAYER2_Z = L.LAYER2_BOTTOM + DROP + GRIP_UP
STICK_PICK_Z = L.TOP + GRIP_UP
STICK_PLACE_Z = L.STICKER_BOTTOM + DROP + GRIP_UP

_BY = dict(L.BOARDS)  # board name -> starting y
_KX = {n: (x, y) for n, x, y in L.STICKERS}
# Where each arm puts its sticker: the left one over the far bearer (x 0.36), the right one over the near bearer
# (x 0.20), both centred on the midline.
STICK_TO = {"left": L.BEARER_XS[1], "right": L.BEARER_XS[0]}


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _pick(x, y, z, open_f, grip_f, hover, yaw, what, move) -> list[Waypoint]:
    return [
        Waypoint(move, (x, y, hover), open_f, f"over the {what}", yaw),
        Waypoint(1.0, (x, y, z + APPROACH), open_f, f"down to just above the {what}", yaw),
        Waypoint(1.5, (x, y, z), open_f, f"slowly down beside the {what}", yaw),
        Waypoint(0.5, (x, y, z), open_f, "settle", yaw),
        Waypoint(1.5, (x, y, z), grip_f, f"grip the {what}", yaw),
        Waypoint(1.2, (x, y, hover), grip_f, f"lift the {what}", yaw),
    ]


def _place(x, y, z, grip_f, open_f, hover, yaw, what, move) -> list[Waypoint]:
    return [
        Waypoint(move, (x, y, hover), grip_f, f"carry the {what} over its place", yaw),
        Waypoint(1.0, (x, y, z + APPROACH), grip_f, "down to 3 cm above", yaw),
        Waypoint(1.5, (x, y, z), grip_f, f"slowly set the {what} down", yaw),
        Waypoint(0.5, (x, y, z), grip_f, "settle", yaw),
        Waypoint(1.5, (x, y, z), open_f, f"let go of the {what}", yaw),
        Waypoint(1.0, (x, y, hover), open_f, "up", yaw),
    ]


def _wait(wps: list[Waypoint], until: float, label: str) -> None:
    """Hold the last pose until ``until`` s on the shared clock (no-op if already past it)."""
    t = _span(wps)
    if until > t + 1e-9:
        w = wps[-1]
        wps.append(Waypoint(until - t, w.pos, w.finger, label, w.yaw))


def _plans() -> dict[str, list[Waypoint]]:
    """Both arms on one clock. Each arm takes the pieces on its own side; the shared stack is visited by one arm
    at a time, and the other arm picks its next piece meanwhile:
      left: inner board -> layer 1 (y +0.05); its sticker -> over the far bearer; outer board -> layer 2 (y +0.05)
      right: inner board -> layer 1 (y -0.05); its sticker -> over the near bearer; outer board -> layer 2"""
    sgn = {"left": 1.0, "right": -1.0}
    yaw_k = {"left": YAW_STICK_L, "right": YAW_STICK_R}
    plan: dict[str, list[Waypoint]] = {"left": [], "right": []}

    def board(arm, which, z_place, move_in, move_to):
        name = f"board_{arm}_{which}"
        y0, s = _BY[name], sgn[arm]
        pick = _pick(L.BOARD_START_X, y0, BOARD_PICK_Z, BOARD_OPEN, BOARD_GRIP, HOVER, YAW_BOARD, f"{which} board",
                     move_in)
        place = _place(L.BOARD_X, s * L.LAYER_Y, z_place, BOARD_GRIP, BOARD_OPEN, HOVER, YAW_BOARD,
                       f"{which} board", move_to)
        return pick, place

    def sticker(arm, move_in, move_to):
        x0, y0 = _KX[f"sticker_{arm}"]
        s, yaw = sgn[arm], yaw_k[arm]  # picked at yaw 0, set down at yaw -+pi/2: a quarter turn in the carry
        pick = _pick(x0 - STICK_OFF, y0, STICK_PICK_Z, STICKER_OPEN, STICKER_GRIP, HOVER, YAW_BOARD, "sticker",
                     move_in)
        place = _place(STICK_TO[arm], s * STICK_OFF, STICK_PLACE_Z, STICKER_GRIP, STICKER_OPEN, HOVER, yaw,
                       "sticker", move_to)
        return pick, place

    lp, rp = plan["left"], plan["right"]
    # 1. Both arms pick their inner board at once; the left lays it on the bearers first.
    pick, place = board("left", "inner", LAYER1_Z, 2.0, 2.0)
    lp += pick + place
    t_left_up = _span(lp)
    pick, place = board("right", "inner", LAYER1_Z, 2.0, 2.0)
    rp += pick
    _wait(rp, t_left_up, "wait for the left board to be down")
    rp += place  # moves in while the left arm moves out to its sticker (both toward +y: TCPs 43 cm apart or more)
    t_right_up = _span(rp)
    # 2. The left arm picks its sticker meanwhile (a quarter turn in the carry) and lays it over the far bearer
    #    once the right arm has left the stack for its own sticker (a 2.5 s move).
    pick, place = sticker("left", 2.5, 2.5)
    lp += pick
    pick_r, place_r = sticker("right", 2.5, 2.5)
    rp += pick_r
    _wait(lp, t_right_up + pick_r[0].duration, "wait for the right arm to leave the stack")
    lp += place
    t_left_up = _span(lp)
    # 3. The left arm goes for its outer board; the right arm's sticker follows 0.5 s later, over the near bearer.
    pick, place = board("left", "outer", LAYER2_Z, 2.0, 2.0)
    lp += pick
    _wait(rp, t_left_up + 0.5, "wait for the left arm to leave the stack")
    rp += place_r
    t_right_up = _span(rp)
    # 4. The left arm lays its outer board on the stickers as the right arm leaves for its outer board; then moves
    #    out, and the right arm lays the last board.
    pick_r, place_r = board("right", "outer", LAYER2_Z, 2.0, 2.0)
    rp += pick_r
    _wait(lp, t_right_up, "wait for the right sticker to be down")
    lp += place
    lp.append(Waypoint(1.5, (L.BOARD_X, 0.25, HOVER), BOARD_OPEN, "out of the way", YAW_BOARD))
    _wait(rp, _span(lp) - 1.5, "wait for the left board to be down")
    rp += place_r
    rp.append(Waypoint(1.5, (L.BOARD_X, -0.25, HOVER), BOARD_OPEN, "out of the way", YAW_BOARD))
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = _plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
