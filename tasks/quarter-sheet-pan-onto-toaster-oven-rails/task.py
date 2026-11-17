"""On the station table, real size: a fixed toaster oven (open front face at x 0.425 m toward the arms; a cavity 344 mm
wide, 245 deep and 170 tall with two pairs of rails on its side walls, the lower pair 55 mm over the floor, and a back
wall), its drop-down door open and flat on the table in front of it (x 0.255 to 0.425, 20 mm tall) and, clear of
the door, a loose quarter sheet pan (9-1/2 x 13 x 1 in, rolled rim, 0.20 kg).

Goal: the pan on the lower rails, level, pushed in until no part of it is in front of the face (the door could
shut; its leading edge is then within 3.7 mm of the back wall), and both grippers 30 mm clear with the pan still for
1 s. The oven's roof and walls leave no room for the hands inside the
cavity: the pan is carried by its two short ends (as a person carries a sheet pan), set on the rails with its trailing
edge still 95 mm outside the front face, let go, and pushed the rest of the way on its near rim with closed fingers
reaching in under the roof (hand pitched forward).

The oven and the pan are mechkit compounds (mechkit.py, next to this file). The arms are read through rigstate.py (the
let-go check). tiltgrip.py wraps ``yam_arena.demo.grip_rotation`` so that the demo's push can be pitched.

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
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (pitched grips for the demo: wraps yam_arena.demo.grip_rotation)
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "Slide the pan onto the toaster oven's lower rails, level and all the way in so the oven door could shut, and take "
    "your hands off it."
)


# -- the parts --------------------------------------------------------------------------------------------------------
def oven_shapes():
    """The oven in its own frame: origin on the table at the middle of the open front face, x into the oven."""
    cw, cd, t = L.CW, L.CD, L.WALL_T
    full_w = cw + 2 * t
    top = L.ROOF_H + L.ROOF_T
    c = L.OVEN_BLACK
    out = [
        mk.box((cd / 2, 0.0, L.FLOOR_H / 2), (cd, full_w, L.FLOOR_H), c),  # floor (its feet are solid)
        mk.box((cd / 2, 0.0, L.ROOF_H + L.ROOF_T / 2), (cd, full_w, L.ROOF_T), c),  # roof
        mk.box((cd + t / 2, 0.0, top / 2), (t, full_w, top), c),  # back wall
    ]
    wall_h = L.ROOF_H - L.FLOOR_H
    for s in (-1, 1):
        out.append(mk.box((cd / 2, s * (cw / 2 + t / 2), L.FLOOR_H + wall_h / 2), (cd, t, wall_h), c))
        for rail_top in (L.RAIL1_TOP, L.RAIL2_TOP):
            out.append(mk.box((cd / 2, s * (cw / 2 - L.RAIL_D / 2), rail_top - L.RAIL_T / 2),
                              (cd, L.RAIL_D, L.RAIL_T), L.RAIL_GREY))
    return out


def pan_shapes():
    """The pan in its own frame: origin at the centre of its underside, x toward its leading edge (into the oven), z up.
    A 4 mm floor, 4 mm walls, and the rolled rim as a solid bead 16 mm wide and 10 mm tall on top of each wall, flush
    with the pan's outer faces."""
    d, w, h, ft, wt, bw, bh = L.PAN_D, L.PAN_W, L.PAN_H, L.PAN_FT, L.PAN_WT, L.BEAD_W, L.BEAD_H
    wall_h = h - bh - ft
    c = L.ALUMINIUM
    out = [mk.box((0.0, 0.0, ft / 2), (d, w, ft), c)]
    for s in (-1, 1):
        out.append(mk.box((s * (d / 2 - wt / 2), 0.0, ft + wall_h / 2), (wt, w, wall_h), c))
        out.append(mk.box((0.0, s * (w / 2 - wt / 2), ft + wall_h / 2), (d - 2 * wt, wt, wall_h), c))
        out.append(mk.box((s * (d / 2 - bw / 2), 0.0, h - bh / 2), (bw, w, bh), c))
        out.append(mk.box((0.0, s * (w / 2 - bw / 2), h - bh / 2), (d - 2 * bw, bw, bh), c))
    return out


OVEN = mk.assembly("oven", origin=(L.OVEN_X, 0.0, L.TOP), shapes=oven_shapes(), color=L.OVEN_BLACK,
                   friction=L.PAN_FRICTION, tight_fit=True)
PAN_PART = mk.assembly(S.PAN, origin=(L.PAN_CX, L.PAN_Y, L.TOP), mass=L.PAN_MASS, shapes=pan_shapes(),
                       color=L.ALUMINIUM, friction=L.PAN_FRICTION, tight_fit=True)
# The open door: a flat slab hinged along the oven's front bottom edge, lying toward the arms (a separate fixed body so
# that its box is not folded into the oven's).
DOOR = mk.assembly("oven_door", origin=(L.OVEN_X - L.DOOR_L / 2, 0.0, L.TOP), color=L.OVEN_BLACK,
                   shapes=[mk.box((0.0, 0.0, L.DOOR_T / 2), (L.DOOR_L, L.CW + 2 * L.WALL_T, L.DOOR_T), L.OVEN_BLACK)],
                   friction=L.PAN_FRICTION)
OBJECTS = [OVEN, DOOR, PAN_PART]


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it knows neither the rails nor a push.
# Offline only (never in the simulator), this model gives the scorer the pan: (1) held by both hands at its two short
# ends (fingers closed across the side beads, hands straight down, 40 mm in from the trailing edge): it moves with the
# hands; (2) let go: it
# drops onto the highest support under it (the lower rails, the upper rails or the table); (3) pushed by a closed,
# pitched hand whose pad corner is on its trailing face: it slides on the rails to where the corner is, stopped by the
# back wall. One hand alone does not hold it level at the rim: it stays where it is.
PAD_THICK = 0.008
LIP_BAND = (0.012, 0.045)  # the TCP over the pan's underside while it grips a rim (pad tips 11.5 mm up, TCP 24.5)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.z = L.PAN_CX, L.PAN_Y, 0.0  # the underside's centre (z above the table top)
        self.held: tuple | None = None  # offsets (pan - mean TCP) while both hands hold it

    def _end_hold(self, side: str, tcp, tips, opening) -> bool:
        sign = 1.0 if side == "left" else -1.0
        a = tips - tcp
        return bool(opening < 0.2 and a[2] < -0.010
                    and abs(float(tcp[0]) - (self.x + L.GRIP_X)) <= 0.020
                    and abs(float(tcp[1]) - (self.y + sign * L.SIDE_Y)) <= 0.006
                    and LIP_BAND[0] <= float(tcp[2]) - L.TOP - self.z <= LIP_BAND[1])

    def _support(self) -> float:
        """The height the pan's underside comes to rest at when let go at (x, y, z)."""
        over_cavity = min(self.x + L.PAN_D / 2, L.BACK_X) - max(self.x - L.PAN_D / 2, L.OVEN_X) >= 0.05
        edge = L.PAN_W / 2 - abs(self.y)
        on_rail = over_cavity and L.PAN_W / 2 - self.y >= L.RAIL_Y + 0.002 and L.PAN_W / 2 + self.y >= L.RAIL_Y + 0.002
        best = 0.0
        for top in (L.RAIL1_TOP, L.RAIL2_TOP):
            if on_rail and self.z >= top - 0.006 and edge <= L.CW / 2 + 1e-9:
                best = max(best, top)
        return best

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        holds = [self._end_hold(side, *r) for side, r in zip(rigstate.ARMS, read)]
        if all(holds):
            tcp = np.mean([r[0] for r in read], axis=0)
            if self.held is None:
                self.held = (self.x - float(tcp[0]), self.y - float(tcp[1]), self.z - (float(tcp[2]) - L.TOP))
            self.x = float(tcp[0]) + self.held[0]
            self.y = float(tcp[1]) + self.held[1]
            self.z = float(tcp[2]) - L.TOP + self.held[2]
            self.x = min(self.x, L.BACK_X - L.PAN_D / 2)
            return
        if self.held is not None:  # let go: it drops onto what is under it
            self.held = None
            self.z = self._support()
        for tcp, tips, opening in read:  # a closed, pitched hand pushing the trailing face
            a = tips - tcp
            n = float(np.linalg.norm(a))
            if opening > 0.2 or n < 1e-6 or a[0] / n < 0.5:
                continue
            a = a / n
            c = np.array([-a[2], 0.0, a[0]])  # the closing axis, perpendicular to the approach, toward +x and up
            corner = tips + PAD_THICK * c
            u = float(corner[2]) - L.TOP - self.z
            if abs(float(corner[1]) - self.y) <= L.PAN_W / 2 and 0.0 <= u <= L.PAN_H:
                face = self.x - L.PAN_D / 2
                if float(corner[0]) > face and float(corner[0]) - face < 0.040:
                    self.x = min(float(corner[0]) + L.PAN_D / 2, L.BACK_X - L.PAN_D / 2)

    def state(self):
        return mk.state_from_poses(PAN_PART, (self.x, self.y, L.TOP + self.z, 0.0, 0.0, 0.0, 1.0), source="offline")


_OFFLINE = OfflineModel()


def no_simulator(pan) -> bool:
    return mk.env() is None and getattr(pan, "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, leading gap mm, trailing out mm, tilt deg]


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the pan's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """On the lower rails, in, level, no part of the pan in front of the face, both grippers 30 mm clear and the
    pan still for 1 s, no latched fall; held 1 s. This runs first every control step, so it also feeds the tracker
    (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    pan = objects.get(S.PAN)
    arms = rigstate.arms()
    if no_simulator(pan):
        _OFFLINE.step(arms)
        pan = _OFFLINE.state()
    result = _TRACKER.update(pan, arms)
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
    row["source"] = getattr(pan, "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["leading_gap_mm"], d["trailing_out_mm"], d["tilt_deg"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the pan is let go and still, on the lower rails and in the oven (and never fell off the table); 1.0
    level and home; 0.5 level, not home, trailing edge within 40 mm of the face; 0.25 otherwise. The score of the latest
    scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below (hands straight down). Finger targets: the joint value is minus half the commanded gap.
PAD = 0.013
OPEN = -0.04695  # 94 mm
GRIP = -(L.BEAD_W - 0.008) / 2  # 8 mm commanded across the 16 mm bead: 20 N a finger
CLOSED = 0.0
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # the push: both close along x
YAW_END = 0.0  # carrying by the short ends: both close along y, across the side beads
TIPS_UP = 0.0115  # the pad tips this far over the pan's underside: 3.5 mm under the bead, 7.5 mm over the floor
HOVER = 0.120
CARRY_U = L.RAIL1_TOP + 0.027  # underside while carried: 27 mm over the lower rails (GPU: hangs 8-10 mm under the
# command, noses down 4.5-6 degrees: the leading edge arrives about 12 mm under its centre)
SET_PRESS = 0.002  # set down commanded this far under the rails' top
STOP_X0 = L.OVEN_X - L.CARRY_STOP  # the trailing edge when the hands let go: 310 mm
RELEASE = -0.013  # 26 mm commanded: 5 mm clear of each face of the bead, and the open carriages stay out of the roof


def grip_z(u):
    """The TCP's height (over the table top) gripping a rim with the pan's underside at ``u``."""
    return u + TIPS_UP + PAD


def end_x(x_trailing):
    """The TCP's x gripping the short ends of a pan whose trailing edge is at ``x_trailing``."""
    return x_trailing + L.END_GRIP


# The push: both hands closed, yaw +-pi/2 (fingers close along x), pitched forward so that the hand reaches in under the
# roof. The pad block's front-lower corner (the outer face of the front pad at the tip) is the contact point.
PUSH_PITCH = math.radians(55.0)
PUSH_Y = 0.11  # each hand pushes the trailing face 110 mm out of the centre line
PUSH_UP = 0.012  # the corner touches the trailing face 12 mm over the pan's underside
OVERTRAVEL = 0.010  # commanded this far into the back wall (the wall stops the pan; the arms trail 3 to 7 mm)
PUSH_SPEED = 0.04  # m/s


def pitched(yaw: float):
    return tiltgrip.tilted(yaw, PUSH_PITCH)


def push_tcp(yaw: float, x_corner: float, up: float, y: float):
    """The TCP of a closed, pitched hand whose front-lower pad corner is at (x_corner, y), ``up`` over the lower rails'
    top (the pan's underside there)."""
    rot = tiltgrip.rotation(yaw, PUSH_PITCH)
    a = rot[:, 2]
    c = rot[:, 0] * (1.0 if rot[0, 0] >= 0 else -1.0)
    corner = np.array([x_corner, y, L.RAIL1_TOP + up])
    return corner - PAD_THICK * c - PAD * a



def _w(d, p, finger, label, yaw):
    return Waypoint(d, (float(p[0]), float(p[1]), float(L.TOP + p[2])), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def hand_moves(side: str, x_end: float = None, push: bool = True):
    """One hand: take the pan's short end (the side rim, 40 mm in from the trailing edge), lift the pan, carry it in
    until its trailing edge is 95 mm outside the face, lower it onto the lower rails, let go; then close the fingers,
    pitch the hand and push the trailing face until the trailing edge is at ``x_end`` (the demo: the leading edge 10 mm
    into the back wall, which stops it); back off and away. ``push=False`` stops after letting go (a counterfactual)."""
    sign = 1.0 if side == "left" else -1.0
    yaw = YAW_L if side == "left" else YAW_R
    y = sign * L.SIDE_Y
    xg = end_x(L.PAN_X0)
    xe = end_x(STOP_X0)
    u0 = 0.0
    ye = YAW_END
    plan = [
        _w(2.0, (xg, y, HOVER), OPEN, "over the pan's short end", ye),
        _w(1.0, (xg, y, grip_z(u0) + 0.030), OPEN, "down to just above its rim", ye),
        _w(1.5, (xg, y, grip_z(u0)), OPEN, "slowly down round the rim", ye),
        _w(0.5, (xg, y, grip_z(u0)), OPEN, "settle", ye),
        _w(1.5, (xg, y, grip_z(u0)), GRIP, "grip the rim", ye),
        _w(0.5, (xg, y, grip_z(u0)), GRIP, "settle the grip", ye),
        _w(2.0, (xg, y, grip_z(CARRY_U)), GRIP, "lift the pan", ye),
        _w(1.3, (end_x(L.PAN_X0 + 0.050), y, grip_z(CARRY_U)), GRIP, "carry it toward the oven", ye),
        _w(4.2, (xe, y, grip_z(CARRY_U)), GRIP, "carry it in over the rails", ye),
        _w(1.5, (xe, y, grip_z(L.RAIL1_TOP - SET_PRESS)), GRIP, "lower it onto the lower rails", ye),
        _w(0.5, (xe, y, grip_z(L.RAIL1_TOP - SET_PRESS)), GRIP, "settle", ye),
        _w(1.5, (xe, y, grip_z(L.RAIL1_TOP - SET_PRESS)), RELEASE, "let go", ye),
        _w(1.0, (xe, y, grip_z(L.RAIL1_TOP) + 0.045), RELEASE, "straight up, clear of the rim", ye),
    ]
    park = (0.22, sign * 0.22, 0.17)
    if not push:
        return plan + [_w(1.5, park, RELEASE, "away", ye)]
    yp = pitched(yaw)
    x_face0 = STOP_X0  # the trailing face before the push
    x_end = L.HOME_X0 + OVERTRAVEL if x_end is None else x_end
    turn = push_tcp(yaw, x_face0 - 0.040, 0.030, sign * L.SIDE_Y)
    start = push_tcp(yaw, x_face0 - 0.040, 0.030, sign * PUSH_Y)
    ready = push_tcp(yaw, x_face0 - 0.010, PUSH_UP, sign * PUSH_Y)
    done = push_tcp(yaw, x_end, PUSH_UP, sign * PUSH_Y)
    back = push_tcp(yaw, x_end - 0.080, PUSH_UP, sign * PUSH_Y)
    return plan + [
        _w(2.5, turn, CLOSED, "close the fingers and pitch the hand forward, back from the trailing face", yp),
        _w(1.0, start, CLOSED, "in to the push line", yp),
        _w(1.5, ready, CLOSED, "down to the face, 10 mm short of it", yp),
        _w(0.5, ready, CLOSED, "settle", yp),
        _w(abs(x_end - (x_face0 - 0.010)) / PUSH_SPEED, done, CLOSED, "push the pan in on the rails", yp),
        _w(0.5, done, CLOSED, "settle", yp),
        _w(1.0, back, CLOSED, "back off 80 mm", yp),
        _w(1.0, park, CLOSED, "away", yaw),
    ]


# Wrong ways (counterfactuals): pushed only part of the way.
VARIANT_OUT = {"short": 0.020, "half": 0.070}  # the trailing edge's distance out of the face where the push stops
VARIANTS = tuple(VARIANT_OUT)


def plans(variant: str = "") -> dict:
    """Both hands' plans for the demo or a counterfactual variant.

    short: pushed until the trailing edge is 20 mm out of the face (0.5).
    half: pushed until it is 70 mm out (0.25)."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    out = {}
    for side in ("left", "right"):
        if variant:
            plan = hand_moves(side, L.OVEN_X - VARIANT_OUT[variant])
        else:
            plan = hand_moves(side)
        out[side] = plan
    return out


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short  pushed until the trailing edge is 20 mm out of the face: 0.5
      half   pushed until the trailing edge is 70 mm out of the face: 0.25"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
