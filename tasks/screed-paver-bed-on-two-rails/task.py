"""On the station table: the bedding bed of a two-paver repair, 200 mm (along x, from the open near edge at x 200 mm to
the far wall at x 415 mm) by 400 mm (along y), inside an edging of three fixed steel flats (far, left, right). Two
fixed 25 x 25 mm steel rails (orange) run along x at y +-112.5 mm, their tops 25 mm above the table: the thickness the
bedding sand must have. The sand is deeper than that (36 mm raked, a mound to 45 mm, a hollow to 19 mm). A loose
aluminium screed board (25 x 50 mm tube, 400 mm long, on edge) lies on two rest blocks in front of the bed's near edge.

The job: take the board off its blocks, lay it across both rails at the far end (x 387.5 mm), pull it 175 mm toward
you with both hands in step so it shaves the sand down to the rail tops, lift it clear and set it back on its blocks,
hands off, bed untouched. The sand is a height model kept by the scorer from the board's measured pose (scoring.py);
the drawn sand is a 20 mm slab under the sole.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

_ONLINE = rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "Take the screed board off its rest blocks, lay it across the two rails at the far end, and pull it toward you "
    "so it shaves the bedding sand down to the rail tops. Then set the board back on its rest blocks and leave the "
    "bed untouched."
)

# -- the parts --------------------------------------------------------------------------------------------------------
BOARD = Box(S.BOARD, pos=L.BOARD_START, size=(L.BD_X, L.BD_L, L.BD_Z), mass=L.BD_MASS, color=L.BD_COLOR)


def _wall(name, x0, x1, y0, y1):
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.WALL_H / 2), size=(x1 - x0, y1 - y0, L.WALL_H),
               color=L.GREY)


def _rail(name, side):
    return Box(name, pos=((L.RAIL_X0 + L.RAIL_X1) / 2, side * L.RAIL_Y, L.TOP + L.RAIL_H / 2),
               size=(L.RAIL_X1 - L.RAIL_X0, L.RAIL_W, L.RAIL_H), color=L.ORANGE)


def _stop(name, side):
    return Box(name, pos=(L.STOP_X, side * L.STOP_Y, L.TOP + L.STOP_H / 2), size=L.STOP_SIZE, color=L.STOP_COLOR)


def _sand(name, y0, y1):
    return Box(name, pos=((L.RAIL_X0 + L.RAIL_X1) / 2, (y0 + y1) / 2, L.TOP + L.SAND_DRAW_H / 2),
               size=(L.RAIL_X1 - L.RAIL_X0, y1 - y0, L.SAND_DRAW_H), color=L.SAND_COLOR)


_RAIL_IN = L.RAIL_Y - L.RAIL_W / 2  # 100 mm: the rails' inner edges
_RAIL_OUT = L.RAIL_Y + L.RAIL_W / 2  # 125 mm
OBJECTS = [
    BOARD,
    _wall("wall_left", L.RAIL_X0, L.WALL_X1, L.INSIDE_Y, L.INSIDE_Y + L.WALL_T),
    _wall("wall_right", L.RAIL_X0, L.WALL_X1, -L.INSIDE_Y - L.WALL_T, -L.INSIDE_Y),
    _wall("wall_far", L.INSIDE_X1, L.WALL_X1, -L.INSIDE_Y, L.INSIDE_Y),
    _rail("rail_left", 1.0),
    _rail("rail_right", -1.0),
    _stop("stop_left", 1.0),
    _stop("stop_right", -1.0),
    _sand("sand_left", _RAIL_OUT, L.INSIDE_Y),
    _sand("sand_mid", -_RAIL_IN, _RAIL_IN),
    _sand("sand_right", -L.INSIDE_Y, -_RAIL_OUT),
]

_TRACKER = S.Tracker()
_EPISODE = {"n": None}


# -- the offline stand-in for the physics ------------------------------------------------------------------------------
# The checker's proxy carries a gripped board rigidly with the TCP and does not stop it on the rails. Offline only
# (never in the simulator) the scorer is given the board lifted to whatever it would rest on: the rails (sole 25 mm,
# anywhere over their length, the board always spans both) or the rest blocks (sole 55 mm, over their 40 mm).
def _support(x: float) -> float:
    if L.RAIL_X0 - L.BD_X / 2 <= x <= L.RAIL_X1:
        return L.RAIL_H
    if abs(x - L.STOP_X) <= L.STOP_SIZE[0] / 2 + L.BD_X / 2 - 0.005:
        return L.STOP_H
    return 0.0


def _offline_board(s):
    """The board state with its sole no lower than what it rests on (a stand-in for contact, offline only)."""
    pos = np.array(s.pos, dtype=float)
    floor = L.TOP + _support(float(pos[0])) + L.BD_Z / 2
    if pos[2] < floor and float(np.max(np.abs(np.asarray(s.lin_vel, dtype=float)))) < 0.5:
        pos[2] = floor
    return SimpleNamespace(pos=pos, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel, ang_vel=s.ang_vel, low=s.low,
                           high=s.high)


# -- scoring hooks ---------------------------------------------------------------------------------------------------
@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the board's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Every sand cell within 3 mm of the rail tops after a bridged pull of at least 150 mm, the board on its rest
    blocks, both hands 30 mm clear and the board still for 1 s, no latched fall, no pressed cell; held 1 s. This runs
    first every control step, so it also feeds the tracker."""
    arms = rigstate.arms()
    board = objects.get(S.BOARD)
    if board is not None and not _ONLINE:
        board = _offline_board(board)
    result = _TRACKER.update({S.BOARD: board}, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    try:
        row["board_quat"] = [round(float(v), 5) for v in board.quat_xyzw]
    except (AttributeError, TypeError, ValueError):
        pass
    if _TRACKER.steps % 10 == 0:
        row["depth_mm"] = np.round(_TRACKER.sand.h * 1000.0, 1).tolist()  # the whole bed every half second
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the board fell off the table, or there was no bridged pull of 150 mm, or fewer than 84 of the 112 cells
    are within 6 mm of 25 mm; 0.25 otherwise; 0.5 every cell within 6 mm of 25 mm; 0.75 within 4 mm and the board
    on its rest blocks; 1.0 within 3 mm, the board stowed, hands 30 mm clear and the board still for 1 s; a pad tip
    (or a board tipped over) in the sand of a finished cell caps it at 0.5. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips
# are 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
GRIP = -(L.BD_X - 0.008) / 2  # 17 mm commanded across the board's 25 mm
OPEN_REL = -(L.BD_X + 0.016) / 2  # 41 mm: just clear of the board when it lets go
YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # both close along x
SIDE = {"left": 1.0, "right": -1.0}
GRIP_Y = 0.180  # 20 mm in from each end of the board, 25 mm clear of the blocks' outer edge
RETREAT_Y = 0.285  # where each hand ends: its finger zone 30 mm or more from the board and the bed
TCP_UP = L.BD_Z - 0.004  # the TCP 4 mm under the board's top: the pads hold the upper 17 mm of its face
CARRY = L.BD_REST_Z + 0.010  # sole height while carrying: 20 mm over the 45 mm mound
LAND = 0.032  # sole 7 mm over the rails before the last slow descent
PRESS = 0.002  # the last descent is commanded 2 mm under the rails' tops: the board rests on them under a light press
PULL_SPEED = 0.0125  # m/s


def _tcp(sole: float) -> float:
    return L.TOP + sole + TCP_UP


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def plan_for(arm: str, variant: str = "") -> list[Waypoint]:
    """One arm's plan: fetch the board at its end, carry it over the bed to the rails' far end, lay it down, pull,
    lift, carry it back and set it on the blocks, let go, step out. Both arms run the same x and z on one clock."""
    y = SIDE[arm] * GRIP_Y
    yaw = YAW[arm]
    ride = L.RAIL_H + (0.005 if variant == "high" else 0.0) - PRESS * (0.0 if variant == "high" else 1.0)
    x_end = L.BD_END_X if variant != "short" else L.BD_START_X - 0.080

    def w(d, x, yy, sole, finger, label):
        return Waypoint(d, (x, yy, _tcp(sole)), finger, label, yaw)

    rest, xs = L.STOP_X, L.BD_START_X
    plan = [
        w(2.0, rest, y, L.BD_REST_Z + 0.045, OPEN, "over the board's end"),
        w(1.0, rest, y, L.BD_REST_Z + 0.030, OPEN, "down to just above the grip"),
        w(1.5, rest, y, L.BD_REST_Z, OPEN, "slowly down beside the board"),
        w(0.5, rest, y, L.BD_REST_Z, OPEN, "settle"),
        w(1.5, rest, y, L.BD_REST_Z, GRIP, "grip the board across its 25 mm"),
        w(0.5, rest, y, L.BD_REST_Z, GRIP, "settle the grip"),
        w(1.5, rest, y, CARRY, GRIP, "lift it 10 mm off the blocks"),
        w(6.0, xs, y, CARRY, GRIP, "carry it over the bed to the rails' far end"),
        w(2.0, xs, y, LAND, GRIP, "down to 7 mm over the rails"),
        w(2.0, xs, y, ride, GRIP, "slowly onto the rails"),
        w(0.5, xs, y, ride, GRIP, "settle"),
        w(abs(xs - x_end) / PULL_SPEED, x_end, y, ride, GRIP, "pull it toward the worker, both ends in step"),
        w(0.5, x_end, y, ride, GRIP, "settle"),
    ]
    if variant == "nostow":
        plan += [w(1.5, x_end, y, ride, OPEN_REL, "let go on the bed"), w(1.5, x_end, SIDE[arm] * RETREAT_Y, ride,
                                                                          OPEN_REL, "step out")]
        return plan
    plan += [
        w(2.0, x_end, y, CARRY, GRIP, "lift it clear of the bed"),
        w(2.0, rest, y, CARRY, GRIP, "carry it to the blocks"),
        w(1.5, rest, y, L.BD_REST_Z - 0.001, GRIP, "lower it onto the blocks"),
        w(0.5, rest, y, L.BD_REST_Z - 0.001, GRIP, "settle"),
        w(1.5, rest, y, L.BD_REST_Z - 0.001, OPEN_REL, "let go"),
        w(2.0, rest, SIDE[arm] * RETREAT_Y, L.BD_REST_Z - 0.001, OPEN_REL, "step out along the board's axis"),
    ]
    return plan


# Wrong ways (counterfactuals): what each does to the score.
VARIANTS = ("short", "high", "nostow")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    return {"left": plan_for("left", variant), "right": plan_for("right", variant)}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short   pulled only 80 mm: no 150 mm bridged pull, cells left at the start depth (0)
      high    pulled 5 mm over the rails: the board skims, never bridged (0)
      nostow  left on the bed after the pull: the bed is level, the board not stowed (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
