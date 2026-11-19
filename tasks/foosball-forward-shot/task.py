"""On the station table, real size: a Best Choice Products 40 in tabletop foosball table (1016 x 572 x 152 mm), its long
side toward the robot, goals at its left and right ends. Eight rods cross the playfield along x: the robot's team (red)
has its handles on the near side, the opponent (blue) on the far side. The ball rests on the playfield just in front
of the robot's three-man forward rod, in the lane between the opponent's goalie and the right-hand goal post.

Objects (mechkit.py next to this file): the table (a fixed mechanism: the cabinet, the playfield, both goals and every
rod but one held still; the robot's forward rod slides along x between its bumpers and turns freely about x, its
players hanging down under their own weight) and the ball (a loose sphere). Nothing is moved or latched by code.

Goal: shoot the ball into the right-hand goal with the forward rod, without spinning the rod and without a hand in the
game. The robot has to slide the rod so its centre forward's foot is behind the ball, then strike with a quick, short
turn of the handle. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=no_align | hold_on (see demo() below).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built)
import rigstate  # noqa: E402
import tiltgrip  # noqa: E402
from scoring import BALL, SLIDE, TABLE, TURN, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Sphere  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("The ball is at your centre forward's feet. Shoot it into the goal on your right with your three-man "
               "rod, without spinning the rod.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _player(x, y_rel, colour):
    """A player hanging from a rod at (x, y_rel) (cabinet frame), as drawn: head above the rod, body, a foot block."""
    rz = L.ROD_Z
    body_top, body_bot = rz + 0.016, rz - L.PLAYER_DROP + L.FOOT[2]
    t, w = L.PLAYER
    return [
        mk.box((x, y_rel, (body_top + body_bot) / 2), (w, t, body_top - body_bot), colour),
        mk.box((x, y_rel, rz - L.PLAYER_DROP + L.FOOT[2] / 2), (L.FOOT[1], L.FOOT[0], L.FOOT[2]), colour),
    ]


def _rod(index, colour, side):
    """Shapes of one rod (cabinet frame): the bar across the cabinet, the handle on its own side, its players."""
    name, _, men, spacing = L.RODS[index]
    y = L.rod_y(index) - L.CAB_CY
    out = [mk.cylinder((L.CAB_W / 2, y, L.ROD_Z), L.ROD_R, L.CAB_W + 0.02, axis="X", color=L.CHROME)]
    hx = -L.HANDLE_L / 2 if side == "near" else L.CAB_W + L.HANDLE_L / 2
    out.append(mk.cylinder((hx, y, L.ROD_Z), L.HANDLE_R, L.HANDLE_L, axis="X", color=L.HANDLE_BLACK))
    for x in L.men_x(men, spacing):
        out += _player(x - L.NEAR_X, y, colour)
    return out


def _cabinet_shapes():
    W, Lc, H, t = L.CAB_W, L.CAB_L, L.CAB_H, L.WALL
    s = [mk.box((W / 2, 0.0, L.FLOOR_Z / 2), (W, Lc, L.FLOOR_Z), L.WOOD),
         mk.box((W / 2, 0.0, L.FLOOR_Z - 0.0005), (W - 2 * t, Lc - 2 * t, 0.001), L.FIELD_GREEN),
         mk.box((t / 2, 0.0, H / 2), (t, Lc, H), L.WOOD), mk.box((W - t / 2, 0.0, H / 2), (t, Lc, H), L.WOOD)]
    # centre line and circle markings
    s.append(mk.box((W / 2, 0.0, L.FLOOR_Z + 0.0002), (W - 2 * t, 0.004, 0.0004), L.LINE_WHITE))
    cx = L.FIELD_CX - L.NEAR_X
    for sign in (-1, 1):  # each end wall, with its goal mouth and the catch box behind it
        ye = sign * (Lc / 2 - t / 2)
        side_w = (W - L.GOAL_W) / 2
        s.append(mk.box((side_w / 2, ye, H / 2), (side_w, t, H), L.WOOD))
        s.append(mk.box((W - side_w / 2, ye, H / 2), (side_w, t, H), L.WOOD))
        top_h = H - L.FLOOR_Z - L.GOAL_H
        s.append(mk.box((cx, ye, H - top_h / 2), (L.GOAL_W, t, top_h), L.WOOD))
        yb = sign * (Lc / 2 + L.GOAL_DEPTH / 2)
        s.append(mk.box((cx, yb, L.FLOOR_Z - 0.01), (L.GOAL_W + 0.02, L.GOAL_DEPTH, 0.02), L.GOAL_BLACK))
        s.append(mk.box((cx, sign * (Lc / 2 + L.GOAL_DEPTH), L.FLOOR_Z + 0.03), (L.GOAL_W + 0.02, 0.006, 0.08),
                        L.GOAL_BLACK))
        for dx in (-1, 1):
            s.append(mk.box((cx + dx * (L.GOAL_W / 2 + 0.006), yb, L.FLOOR_Z + 0.03), (0.006, L.GOAL_DEPTH, 0.08),
                            L.GOAL_BLACK))
    for i, (name, side, _, _) in enumerate(L.RODS):  # every rod but the shooter is held still
        if name != L.SHOOTER:
            s += _rod(i, L.RED if side == "near" else L.BLUE, side)
    return s


def _table():
    i = L.rod_index(L.SHOOTER)
    y = L.rod_y(i) - L.CAB_CY
    # the forward rod: a short collar on the slide joint (inside the near wall), the rod with its players on the turn
    collar = [mk.cylinder((L.WALL + 0.012, y, L.ROD_Z), L.ROD_R + 0.004, 0.01, axis="X", color=L.CHROME)]
    turn = mk.spindle(TURN, "X", (L.CAB_W / 2, y, L.ROD_Z), _rod(i, L.RED, "near"), mass=0.25,
                      damping=0.0005, friction=L.ROD_FRICTION, parent=SLIDE, color=L.RED)
    slide = mk.slider(SLIDE, "X", collar, lower=-L.SLIDE, upper=L.SLIDE, anchor=(L.WALL + 0.012, y, L.ROD_Z),
                      mass=0.02, damping=2.0, friction=L.SLIDE_FRICTION)
    return mk.assembly(TABLE, (L.NEAR_X, L.CAB_CY, L.TOP), _cabinet_shapes(), joints=[slide, turn], color=L.WOOD,
                       friction=L.TABLE_FRICTION)


TABLE_SPEC = _table()
BALL_SPEC = Sphere(BALL, pos=L.ball_start(), color=L.BALL_COLOUR, mass=L.M_BALL, radius=L.BALL_R)
OBJECTS = [TABLE_SPEC, BALL_SPEC]


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


def _r(v, n: int = 5) -> list:
    return [round(float(x), n) for x in np.asarray(v, dtype=float).ravel()]


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """A goal in the right-hand goal, no spin, no hand in the game, the handle let go; held 1 s. Runs every control
    step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=_r(arms["left"].tcp), right_tcp=_r(arms["right"].tcp),
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 goal and let go; 0.75 goal still holding; 0.5 on target; 0.25 150 mm toward goal; else 0."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, the right arm only. A straight-down grip on the forward rod's handle, the pads closing on it from either
# side (along y), near its outer end, 85 mm out from the cabinet. The hand slides the rod 38 mm out so the centre forward's foot is
# behind the ball, winds the rod back 25 deg (feet up toward the robot's left), then turns it 45 deg the other way in
# 0.5 s, so the foot swings down through the ball and drives it toward the right-hand goal past the goalie's foot.
# The hand turns about the rod's axis (the TCP sits 8 mm above it, so it follows the small circle that point
# makes). It turns the rod back to hanging straight, lets go and lifts away.
OPEN = -0.04695
GRIP = -(L.HANDLE_R - 0.006)  # 6 mm squeeze on the 32 mm rubber handle
GRIP_X = L.NEAR_X - 0.085  # along the handle, 85 mm out from the cabinet (the hand clears its wall after the slide)
TCP_ABOVE_AXIS = 0.008  # the TCP sits this far above the rod's axis when the pads straddle the handle
WIND_DEG = 25.0
STRIKE_DEG = -20.0
STRIKE_S = 0.50
VARIANTS = ("no_align", "hold_on")


def tcp(slide: float, turn_deg: float = 0.0) -> np.ndarray:
    """The TCP holding the handle with the rod slid ``slide`` and turned ``turn_deg`` (about +x)."""
    a = math.radians(turn_deg)
    y = L.rod_y(L.rod_index(L.SHOOTER))
    return np.array([GRIP_X + slide, y - TCP_ABOVE_AXIS * math.sin(a), L.TOP + L.ROD_Z + TCP_ABOVE_AXIS * math.cos(a)])


def _w(duration, pos, finger, label, turn_deg=0.0) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label,
                    tiltgrip.tilted(0.0, math.radians(turn_deg)))


def plan(align: bool = True, let_go: bool = True) -> list[Waypoint]:
    s = L.BALL_OFFSET_X if align else 0.0
    w = [
        _w(2.5, tcp(0.0) + [0.0, 0.0, 0.06], OPEN, "over the forward rod's handle"),
        _w(1.2, tcp(0.0), OPEN, "down: the pads either side of the handle"),
        _w(0.8, tcp(0.0), GRIP, "close on the handle"),
        _w(1.2, tcp(s), GRIP, "slide the rod: the centre forward behind the ball"),
        _w(1.0, tcp(s, WIND_DEG), GRIP, "wind back", WIND_DEG),
        _w(0.4, tcp(s, WIND_DEG), GRIP, "settle", WIND_DEG),
        _w(STRIKE_S, tcp(s, STRIKE_DEG), GRIP, "strike: turn the rod through the ball", STRIKE_DEG),
        _w(0.6, tcp(s, STRIKE_DEG), GRIP, "follow through", STRIKE_DEG),
        _w(1.0, tcp(s), GRIP, "back to hanging"),
    ]
    if not let_go:
        return w + [_w(3.0, tcp(s), GRIP, "keep holding the handle")]
    return w + [
        _w(0.8, tcp(s), OPEN, "let go"),
        _w(1.0, tcp(s) + [0.0, 0.0, 0.07], OPEN, "straight up"),
        _w(1.5, (0.18, -0.30, L.TOP + 0.20), OPEN, "away"),
        _w(2.5, (0.18, -0.30, L.TOP + 0.20), OPEN, "hold"),
    ]


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "no_align":  # struck without sliding the rod first: the foot misses the ball
        return {"right": plan(align=False)}
    if variant == "hold_on":  # a goal, the handle still held at the end
        return {"right": plan(let_go=False)}
    return {"right": plan()}


def demo():
    """The right arm (the left stays parked). DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_align  struck without lining the forward up with the ball (0: the foot misses)
      hold_on   a goal, the handle still held at the end (0.75)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
