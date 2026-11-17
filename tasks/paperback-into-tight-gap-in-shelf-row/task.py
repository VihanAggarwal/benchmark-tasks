"""On the station table, the top shelf of a small open bookcase in front of the left arm: a 12 mm shelf board between
two 18 mm side panels with a 12 mm back panel (all fixed, 240 mm high, nothing above the shelf). Six upright paperbacks
(A to F, 105 x 148 mm, 25 mm thick, 0.16 kg) fill the shelf from panel to panel, spines toward the robot, except for a
40 mm gap between C and D. On the table to the left of the case, in front of its end panel, stands the new book: a
taller, thicker paperback (105 x 185 mm, 30 mm thick, 0.26 kg), spine toward the robot. Put it into the gap, square,
with its spine on the row's spine line, leave all six neighbours upright and in place, and take both hands off.

The scorer reads the arms (pad-tip points and openings) through rigstate.py, which wraps the embodiment's step and reset
from this file's import, only to check that the grippers have let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Trace: TASK_TRACE=<file> (one JSON line per step).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker, clearances, gap_mid_y, spine_line, spine_x, tilt_deg, yaw_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# The scorer's own last second (hands off, every book still) is on top of this 1 s.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Put the loose red paperback upright into the gap in the row on the shelf, spine out and square, with its spine "
    "level with the other spines. Leave every other book upright and where it is, and take both hands off."
)

# -- the scene -----------------------------------------------------------------------------------------------------
_PW = L.CASE_X1 - L.SHELF_X0


def _fixed(name, x0, x1, y, size_y, z0, h, color):
    return Box(name, pos=((x0 + x1) / 2, y, L.TOP + z0 + h / 2), size=(x1 - x0, size_y, h), color=color)


SHELF = _fixed("shelf_board", L.SHELF_X0, L.SHELF_X1, L.Y0, 2 * L.PANEL_Y_IN, 0.0, L.SHELF_T, L.SHELF_WHITE)
SIDE_R = _fixed("case_side_r", L.SHELF_X0, L.CASE_X1, L.Y0 - L.PANEL_Y_IN - L.PANEL_T / 2, L.PANEL_T, 0.0, L.PANEL_H,
                L.CASE_PINE)
SIDE_L = _fixed("case_side_l", L.SHELF_X0, L.CASE_X1, L.Y0 + L.PANEL_Y_IN + L.PANEL_T / 2, L.PANEL_T, 0.0, L.PANEL_H,
                L.CASE_PINE)
BACK = _fixed("case_back", L.SHELF_X1, L.CASE_X1, L.Y0, 2 * L.PANEL_Y_IN, 0.0, L.PANEL_H, L.CASE_PINE)


def _book(name: str) -> Box:
    if name == L.NEW:
        x, y = L.NEW_START
        return Box(name, pos=(x, y, L.TOP + L.NEW_H / 2), size=L.SIZE[name], mass=L.NEW_MASS, color=L.NEW_RED)
    x, y = L.NB_START[name]
    return Box(name, pos=(x, y, L.SHELF_TOP + L.NB_H / 2), size=L.SIZE[name], mass=L.NB_MASS,
               color=L.BOOK_COLOURS[name])


BOOKS = [_book(n) for n in L.BOOKS]
OBJECTS = [SHELF, SIDE_R, SIDE_L, BACK, *BOOKS]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the books teleporting back)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided. A trace that
    cannot be written is skipped (it must never break a run)."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def _states(objects) -> dict:
    return {n: objects[n] for n in L.BOOKS}


def success(objects) -> bool:
    """1.0 now (the new book seated square in the gap, the row upright and in place, no book ever fallen, both grippers
    30 mm clear and every book still for the last second), held SUCCESS_HOLD_S = 1 s. This runs first every control
    step, so it also feeds the tracker with the arms rigstate read for this step."""
    books = _states(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(books, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        n = books[L.NEW]
        row["new_pos_mm"] = [round(float(v) * 1000, 1) for v in n.pos]
        if arms is not None:
            tcp = arms["left"].tcp
            row["left_tcp_mm"] = [round(float(v) * 1000, 1) for v in tcp]
            row["left_opening"] = round(float(arms["left"].opening), 3)
            row["book_minus_tcp_y_mm"] = round((float(n.pos[1]) - float(tcp[1])) * 1000, 1)
            row["hands_gap_mm"] = round(float(rigstate.hands_gap(arms, __import__("scoring").book_distance(books)))
                                        * 1000, 1)
        else:
            row["rigstate_error"] = rigstate.error()
    except (AttributeError, KeyError, TypeError, ValueError):
        row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any book ever tilted more than 45 degrees, a neighbour ends off the shelf, or unless both grippers are 30 mm
    clear and every book still for the last second. Then 1.0: the new book seated (underside on the shelf within 3 mm,
    centre within 3 mm of the gap's centre line, every bottom corner 2 mm or more clear of C and D, tilt under 3
    degrees, turned under 3 degrees, spine within 5 mm of the
    row's spine line) and A to F upright (end tilt under 5 degrees, never over 20, each within 6 mm of its place); 0.5:
    in the gap (all four underside corners between C and D, underside within 3 mm of the shelf, tilt under 10 degrees,
    turned under 15, spine within 40 mm of the line) and the row upright; 0.25: in the gap with the row disturbed (each of
    A to F under 20 degrees at the end and within 20 mm of its place); 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout. TCP points; the pad tips are 13 mm below. Finger targets: the
# joint value is minus half the commanded gap; command 8 mm narrower than the part (the book is 30 mm thick).
PAD = 0.013
OPEN = -0.04695  # 94 mm
GRIP = -(L.NEW_T - 0.008) / 2  # 22 mm commanded across the 30 mm book
RELEASE = -0.022  # 44 mm: 7 mm clear of each cover when it lets go
YAW = 0.0  # the fingers close along y (across the book's thickness); the left arm's base line is the gap's line
# The pads grip the top of the book, above the neighbours: pad tips 10 mm over their tops, so the pads cover the book's
# top 27 mm of its 185 (TCP 171 mm over its underside; the pad's flat face runs from 9 mm above the TCP to the tip).
TCP_UP = L.NB_H + 0.010 + PAD  # 171 mm: TCP height over the book's underside
GRIP_Z = L.TOP + TCP_UP  # at the start, the book on the table
CARRY_CLEAR = 0.012  # carried: its underside 12 mm over the shelf top (the arm sags 3 to 7 mm under the book, and its front edge must clear the shelf board's front edge)
CARRY_Z = L.SHELF_TOP + CARRY_CLEAR + TCP_UP
PRESS = 0.003  # set down 3 mm lower than it rests, so it carries no weight on the arm when the fingers open
SET_Z = L.SHELF_TOP - PRESS + TCP_UP
APPROACH = 0.020  # the final descent starts this far above the grip height and takes 1.5 s (the open pads are 94 mm
#                   apart, so they pass either side of the 30 mm book from any height)
OVERTRAVEL = 0.004  # the commanded end is this far past where the book's back meets the back panel
Y_TRIM = 0.0  # GPU: if TASK_TRACE shows book_minus_tcp_y_mm settling at a constant while carried, put its negative here
CARRY_X = 0.170  # while carried across: the book's 105 mm span ends 12 mm short of the case's front
PUSH_SPEED = 0.035  # m/s along x into the gap
HOVER_Z = L.TOP + 0.205  # travel height over the book: the open pads' tips 192 mm up, 7 mm over its top
FRONT_X = 0.210  # where the hand first goes: in front of the case's end panels (their front edges are at x 235 mm)
BACK_X = L.XS - 0.095  # where the open hand backs off to: its axis 95 mm in front of the spines


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, z), finger, label, YAW)


def plan() -> list[Waypoint]:
    x0, y0 = L.NEW_START
    xe = L.NEW_END_X + OVERTRAVEL
    ye = L.Y0 + Y_TRIM
    push = (xe - CARRY_X) / PUSH_SPEED
    return [
        _w(2.0, FRONT_X, L.Y0, HOVER_Z, OPEN, "out in front of the case, clear of its end panel"),
        _w(1.5, x0, y0, HOVER_Z, OPEN, "over the new book, the open pads 7 mm over its top"),
        _w(1.0, x0, y0, GRIP_Z + APPROACH, OPEN, "down either side of its top"),
        _w(1.5, x0, y0, GRIP_Z, OPEN, "slowly down round its top"),
        _w(0.5, x0, y0, GRIP_Z, OPEN, "settle"),
        _w(1.5, x0, y0, GRIP_Z, GRIP, "grip the book across its thickness"),
        _w(0.5, x0, y0, GRIP_Z, GRIP, "settle the grip"),
        _w(1.5, x0, y0, CARRY_Z, GRIP, "lift it clear of the table, up to shelf height"),
        _w(3.0, CARRY_X, ye, CARRY_Z, GRIP, "carry it across, in line with the gap"),
        _w(0.5, CARRY_X, ye, CARRY_Z, GRIP, "settle in line"),
        _w(push, xe, ye, CARRY_Z, GRIP, "slide it into the gap until its back meets the back panel"),
        _w(0.5, xe, ye, CARRY_Z, GRIP, "settle"),
        _w(2.0, xe, ye, SET_Z, GRIP, "lower it onto the shelf"),
        _w(0.5, xe, ye, SET_Z, GRIP, "settle"),
        _w(1.5, xe, ye, SET_Z, RELEASE, "let go"),
        _w(0.5, xe, ye, SET_Z, RELEASE, "settle"),
        _w(0.5, xe, ye, SET_Z + 0.006, RELEASE, "up 6 mm, the pads clear of the covers"),
        _w(2.0, BACK_X, ye, SET_Z + 0.006, RELEASE, "back off along x, out of the case"),
    ]


def demo():
    """The left arm does the job; the right arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the
    ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    p = {"left": plan()}
    p.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return p
