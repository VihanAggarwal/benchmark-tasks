"""On the station table (the nursery's counter, real size): a two-well bottle warmer (110 x 320 x 60 mm, cream
plastic, two round wells 72 mm across and 50 mm deep, one push button per well on its top) stands against a
tiled splashback. In its +y well stands a child's feeding bottle (260 mL, 62 mm across, 135 mm tall, 0.20 kg, pink
collar) that is already warming. A second child's bottle (blue collar) waits on a bottle tray on the counter, 250 mm
to the right of the warmer.

Goal: the second bottle standing upright on the floor of the free (-y) well, that well started (its button pushed in
once, while the bottle is seated; a second push turns it off), the first bottle never moved, hands off. The wells are 5 mm wider each side than the
bottle; the open gripper's pads are 94 mm apart, so the bottle is carried gripped by its upper body, high enough for
the pads to clear the warmer's top.

The warmer, the tray, the splashback and the bottles are mechkit compounds (mechkit.py, next to this file); the warmer
is a fixed articulation whose two buttons are sprung sliders. No software rule. The arms are read through rigstate.py
(the let-go check).

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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import BOTTLE, NEIGHBOUR, WARMER  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Put the bottle from the tray into the warmer's free well and start that well. "
               "Leave the bottle already warming undisturbed.")


# -- the parts --------------------------------------------------------------------------------------------------------
def warmer_shapes():
    out = [mk.box((0, 0, L.FLOOR / 2), (L.W_LX, L.W_LY, L.FLOOR), L.CREAM)]
    for wy in (L.WELL_Y, -L.WELL_Y):  # 16 turned wall boxes round each well, 50 mm tall
        for (dx, dy), tang, yaw in L.ring_segments():
            out.append(mk.box((dx, wy + dy, L.FLOOR + L.WELL_DEPTH / 2),
                              (L.WELL_OUT - L.WELL_IN, tang, L.WELL_DEPTH), L.CREAM, ypr=(yaw, 0.0, 0.0)))
    holes = [(0.0, wy, 0.076, 0.076) for wy in (L.WELL_Y, -L.WELL_Y)]
    out += list(mk.plate_with_holes((0, 0, L.W_H - L.PLATE_T / 2), (L.W_LX, L.W_LY), holes, L.PLATE_T))
    h = L.W_H - L.PLATE_T - L.FLOOR
    z = L.FLOOR + h / 2
    out += [mk.box((-(L.W_LX - L.SKIN) / 2, 0, z), (L.SKIN, L.W_LY, h), L.CREAM),
            mk.box(((L.W_LX - L.SKIN) / 2, 0, z), (L.SKIN, L.W_LY, h), L.CREAM),
            mk.box((0, -(L.W_LY - L.SKIN) / 2, z), (L.W_LX - 2 * L.SKIN, L.SKIN, h), L.CREAM),
            mk.box((0, (L.W_LY - L.SKIN) / 2, z), (L.W_LX - 2 * L.SKIN, L.SKIN, h), L.CREAM)]
    return out


def button(name, wy):
    cap = mk.cylinder((L.BTN_DX, wy, L.BTN_TOP - L.BTN_LEN / 2), L.BTN_R, L.BTN_LEN, "Z", color=L.BTN_GREEN)
    return mk.slider(name, "-Z", [cap], upper=L.BTN_TRAVEL, lower=0.0, mass=0.01, spring=L.BTN_SPRING,
                     damping=L.BTN_DAMP)


def bottle_shapes(collar):
    return [mk.cylinder((0, 0, L.B_BODY_H / 2), L.B_R, L.B_BODY_H, color=L.B_BODY),
            mk.cylinder((0, 0, (L.B_COLLAR_Z0 + L.B_COLLAR_Z1) / 2), L.B_R, L.B_COLLAR_Z1 - L.B_COLLAR_Z0,
                        color=collar),
            mk.cylinder((0, 0, (L.B_COLLAR_Z1 + L.B_H) / 2), L.B_HOOD_R, L.B_H - L.B_COLLAR_Z1, color=L.B_HOOD)]


WARMER_PART = mk.assembly(WARMER, origin=(L.WX, L.WY, L.TOP), shapes=warmer_shapes(), color=L.CREAM,
                          joints=[button("btn_a", L.BTN_Y), button("btn_b", -L.BTN_Y)], tight_fit=True)
TRAY = mk.assembly("tray", origin=(L.TRAY_X, L.TRAY_Y, L.TOP), color=L.TRAY_GREY, tight_fit=True,
                   shapes=mk.open_box((0, 0, L.TRAY_H / 2), (L.TRAY_L, L.TRAY_L, L.TRAY_H), wall=L.TRAY_WALL,
                                      floor=L.TRAY_FLOOR, color=L.TRAY_GREY))
SPLASHBACK = mk.assembly("splashback", origin=((L.WALL_X0 + L.WALL_X1) / 2, (L.WALL_Y0 + L.WALL_Y1) / 2, L.TOP),
                         color=L.TILE,
                         shapes=[mk.box((0, 0, L.WALL_H / 2), (L.WALL_X1 - L.WALL_X0, L.WALL_Y1 - L.WALL_Y0,
                                                               L.WALL_H), L.TILE)])
NEIGHBOUR_PART = mk.assembly(NEIGHBOUR, origin=(L.A_X0, L.A_Y0, L.TOP + L.FLOOR), mass=L.B_MASS,
                             shapes=bottle_shapes(L.COLLAR_A), color=L.B_BODY, tight_fit=True)
BOTTLE_PART = mk.assembly(BOTTLE, origin=(L.BOT_X0, L.BOT_Y0, L.TOP + L.BOT_Z0), mass=L.B_MASS,
                          shapes=bottle_shapes(L.COLLAR_B), color=L.B_BODY, tight_fit=True)
OBJECTS = [WARMER_PART, TRAY, SPLASHBACK, NEIGHBOUR_PART, BOTTLE_PART]
SPECS = {NEIGHBOUR: NEIGHBOUR_PART, BOTTLE: BOTTLE_PART, WARMER: WARMER_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows the warmer only as its bounding box, so a released bottle would stand on its top. Offline
# only (never in the simulator) this model gives the scorer: (1) the bottle carried by a gripper that closed on it
# (pads over its upper body), dropped straight down on release onto the well's floor (within 5 mm of its axis), the
# warmer's top or the tray; (2) the free well's button pushed down by a closed gripper's pad block whose tips pass the
# button's resting top (and the neighbour's likewise); (3) the neighbour never moves.
PAD_TIP = 0.013
PAD_HALF_X = 0.008  # a closed pad block across the closing direction (yaw pi/2: along x): 16 mm


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = np.array([L.BOT_X0, L.BOT_Y0, L.BOT_Z0])
        self.held: str | None = None
        self.rel = np.zeros(3)
        self.btn = {"btn_a": 0.0, "btn_b": 0.0}

    def _support(self) -> float:
        x, y = float(self.pos[0]), float(self.pos[1])
        if math.hypot(x - L.WELL_B[0], y - L.WELL_B[1]) <= L.WELL_IN - L.B_R + 0.001:
            return L.FLOOR
        if L.W_X0 <= x <= L.W_X1 and L.W_Y0 <= y <= L.W_Y1:
            return L.W_H
        if abs(x - L.TRAY_X) <= L.TRAY_L / 2 - L.TRAY_WALL and abs(y - L.TRAY_Y) <= L.TRAY_L / 2 - L.TRAY_WALL:
            return L.TRAY_FLOOR
        return 0.0

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        sides = dict(zip(rigstate.ARMS, read))
        if self.held is not None:
            tcp, tips, opening = sides[self.held]
            if opening > 0.70:
                self.held = None
                self.pos[2] = self._support()
            else:
                self.pos = (tcp - np.array([0.0, 0.0, L.TOP])) + self.rel
        else:
            for side, (tcp, tips, opening) in sides.items():
                p = tcp - np.array([0.0, 0.0, L.TOP])
                above = (float(tips[2]) - L.TOP) - self.pos[2]
                if (opening < 0.62 and math.hypot(p[0] - self.pos[0], p[1] - self.pos[1]) <= 0.012
                        and 0.045 <= above <= 0.100):
                    self.held, self.rel = side, self.pos - p
                    break
        pressed = {"btn_a": 0.0, "btn_b": 0.0}
        for side, (tcp, tips, opening) in sides.items():
            if opening > 0.2:
                continue
            tip_z = float(tips[2]) - L.TOP
            if abs(float(tcp[0]) - (L.WX + L.BTN_DX)) > L.BTN_R + PAD_HALF_X:
                continue
            for key, wy in (("btn_a", L.BTN_Y), ("btn_b", -L.BTN_Y)):
                if abs(float(tcp[1]) - wy) <= L.BTN_R + rigstate.FINGER_HALF_WIDTH:
                    depth = L.BTN_TOP - tip_z
                    pressed[key] = max(pressed[key], min(max(depth, 0.0), L.BTN_TRAVEL))
        self.btn = pressed

    def states(self) -> dict:
        nb = mk.state_from_poses(NEIGHBOUR_PART, (L.A_X0, L.A_Y0, L.TOP + L.FLOOR, 0.0, 0.0, 0.0, 1.0),
                                 source="offline")
        bt = mk.state_from_poses(BOTTLE_PART, (self.pos[0], self.pos[1], L.TOP + self.pos[2], 0.0, 0.0, 0.0, 1.0),
                                 source="offline")
        wm = mk.state_from_poses(WARMER_PART, (L.WX, L.WY, L.TOP, 0.0, 0.0, 0.0, 1.0), joints=dict(self.btn),
                                 source="offline")
        return {NEIGHBOUR: nb, BOTTLE: bt, WARMER: wm}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(NEIGHBOUR), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: every key part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The loose bottle seated upright on the free well's floor, that well on (an odd number of pushes) and never
    pushed while the bottle was not seated, the neighbour never disturbed, no latch, both grippers 30 mm clear and both bottles and the buttons still for 1 s;
    held 1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline
    stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    if no_simulator(parts):
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(BOTTLE), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if either bottle fell off the counter or tilted over 30 degrees, the neighbour left its well, or the loose
    bottle is neither seated nor on the warmer; 1.0 seated, started, both grippers clear, all still; 0.75 the same
    not yet clear and still; 0.5 seated, not started; 0.25 resting on the warmer (or leaning in the free well), hands off and still; capped at 0.5 (0.25 if not started)
    if the neighbour was ever disturbed or the well was pushed while the bottle was not seated. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
# The right arm does everything: the tray, the free well and its button lie in its column.
OPEN = -0.04695  # 94 mm: 16 mm each side of the 62 mm bottle
GRIP = -(2 * L.B_R - 0.008) / 2  # 54 mm commanded across the 62 mm bottle
CLOSED = 0.0
YAW = math.pi / 2  # fingers close along x: the pads stay off the neighbour's side and out of the well's walls
APPROACH = 0.030

GRIP_ABOVE_BASE = 0.103  # TCP 103 mm over the bottle's base: pad tips 90 mm up its side (pads 90 to 128 mm up), so the
# cap's top (135 mm) is 32 mm over the TCP, under the gripper body's proxy (it reaches 35 mm above the TCP); seated
# the pad tips are 40 mm over the warmer's top
PICK_TCP = L.BOT_Z0 + GRIP_ABOVE_BASE  # 92 mm
HOVER = 0.172  # pad tips 19 mm over the bottle's top (139 mm up on the tray)
CARRY_TCP = 0.173  # the bottle's base 70 mm up: 10 mm over the warmer's top
SEAT_TCP = L.FLOOR + GRIP_ABOVE_BASE + 0.002  # base 2 mm over the floor
AWAY = (L.WELL_B[0], -0.250)  # beside the tray, 195 mm from the well's axis
AWAY_Z = 0.150
BTN_XY = (L.WX + L.BTN_DX, -L.BTN_Y)  # the free well's button, pushed from above by the closed block
OVER_BTN_Z = L.BTN_TOP + 0.030 + PAD_TIP  # pad tips 30 mm over the button's rest top
NEAR_BTN_Z = L.BTN_TOP + 0.010 + PAD_TIP  # pad tips 10 mm over it: the slow descent starts here
# The pad tips are COMMANDED 9 mm under the button's rest top (3 mm under the warmer's top, which the block's 27 mm
# width meets beside the 18 mm cap). The arm gives under the spring: on the GPU (conveyor task) it stalled 3.3 mm short
# against 2.9 N (about 0.9 N/mm), so the cap goes in 5 mm (its stop) at 0.9 N/mm and 4.7 mm even at 0.43 N/mm.
PRESS = 0.009
PRESS_HOLD_S = 1.0
HOVER_BTN = 0.150
END_Z = 0.140  # TCP over the warmer's top when done: pad tips 127 mm up, finger zones well over 30 mm clear


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, YAW)


def pick_moves():
    px, py = L.BOT_X0, L.BOT_Y0
    return [
        _w(2.0, px, py, HOVER, OPEN, "over the bottle, pads clear of its cap"),
        _w(1.0, px, py, PICK_TCP + APPROACH, OPEN, "down round the bottle"),
        _w(1.5, px, py, PICK_TCP, OPEN, "slowly down to the grip height"),
        _w(0.5, px, py, PICK_TCP, OPEN, "settle"),
        _w(1.5, px, py, PICK_TCP, GRIP, "grip the bottle across its 62 mm"),
        _w(2.0, px, py, CARRY_TCP, GRIP, "lift it clear of the tray and the warmer's height"),
    ]


def seat_moves(dy=0.0, seat_tcp=SEAT_TCP):
    qx, qy = L.WELL_B[0], L.WELL_B[1] + dy
    return [
        _w(2.5, qx, qy, CARRY_TCP, GRIP, "carry it over the free well"),
        _w(1.0, qx, qy, seat_tcp + 0.025, GRIP, "down to 25 mm over the floor"),
        _w(2.0, qx, qy, seat_tcp, GRIP, "slowly down into the well"),
        _w(0.5, qx, qy, seat_tcp, GRIP, "settle"),
        _w(1.5, qx, qy, seat_tcp, OPEN, "let go: it stands"),
        _w(1.0, qx, qy, AWAY_Z, OPEN, "straight up, pads either side"),
    ]


def away_moves(finger=CLOSED):
    return [_w(2.0, AWAY[0], AWAY[1], AWAY_Z, finger, "aside over the tray, closing the fingers into a block")]


def start_moves(press=True, depth=PRESS):
    x, y = BTN_XY
    plan = [
        _w(2.5, x, y, OVER_BTN_Z, CLOSED, "over the free well's button, closed block, tips 30 mm over it"),
        _w(0.5, x, y, OVER_BTN_Z, CLOSED, "settle"),
        _w(1.0, x, y, NEAR_BTN_Z, CLOSED, "down to 10 mm over the button"),
    ]
    if press:
        plan += [
            _w(1.5, x, y, L.BTN_TOP - depth + PAD_TIP, CLOSED, "slowly down on the button, pushing it in"),
            _w(PRESS_HOLD_S, x, y, L.BTN_TOP - depth + PAD_TIP, CLOSED, "hold it in"),
            _w(1.0, x, y, NEAR_BTN_Z, CLOSED, "up off the button"),
        ]
    plan += [_w(1.5, x, y, END_Z, CLOSED, "up and clear")]
    return plan


VARIANTS = ("no_start", "press_first", "short_press")


def plans(variant: str = "") -> dict:
    """The right arm's plan for the demo or a counterfactual variant (the left arm holds its ready pose).

    Order of work: the bottle from the tray to the free well, let go, aside, the free well's button pushed down, up
    and clear."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "press_first":  # the button pushed before the bottle is in: a dry start caps the score
        return {"right": [_w(2.0, 0.20, -0.250, HOVER_BTN, CLOSED, "closed block aside first")]
                + start_moves() + [_w(2.0, L.BOT_X0, L.BOT_Y0, HOVER, OPEN, "over the bottle")]
                + pick_moves()[1:] + seat_moves() + away_moves(OPEN)}
    right = pick_moves() + seat_moves() + away_moves()
    if variant == "no_start":
        return {"right": right}
    if variant == "short_press":  # the pad tips stop 2 mm into the button's travel (it needs 3)
        return {"right": right + start_moves(depth=0.002)}
    return {"right": right + start_moves()}


def demo():
    """The right arm on one clock; the left holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_start     the bottle seated, the button never pushed (0.5)
      press_first  the button pushed first (a dry start, capped), then the bottle seated (0.5)
      short_press  the button pushed only 2 mm (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
