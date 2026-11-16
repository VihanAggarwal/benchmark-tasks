"""On the station table, real size where a real figure exists: the side wall of a dish rack (a round 10 mm top rail at
z 135 mm over two 6 mm lower wires, on two end posts, all fixed) stands across the table at x 0.33 m, centred on
y -0.11 m, with a draining tray and two plates on edge behind it. In front of it, at (x 0.20, y -0.21), stands a slim
cutlery caddy (76 x 18 x 100 mm, three 22 mm slots, 0.06 kg) with three 120 mm forks standing tines down in its slots.
Its back wall carries two hooks: each an inverted U (the wall's own tab, a bridge, a rear leg with a flared tip) with a
18 mm gap over the 10 mm rail.

Goal: the caddy hung on the rail by both hooks (the bridges resting on the rail, the back wall leaning on the lower
wires, the base 14 mm over the table), still square, all three forks still in it, hands off. Lowered with the hooks
a few millimetres off, a leg lands on top of the rail or the caddy hangs on one hook and drops at the first knock.

The caddy, the forks and the rack are mechkit compounds (mechkit.py, next to this file). No software rules: every
part is a plain rigid body. The arms are read through rigstate.py (the let-go check).

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
from scoring import CADDY, FORKS  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Hang the caddy on the rack's rail by both hooks, and let go."


# -- the parts --------------------------------------------------------------------------------------------------------
def caddy_shapes():
    """The caddy in its own frame (origin: middle of its footprint, at its base's underside; +x the back wall)."""
    c, t, d, w, h = L.CADDY_BLUE, L.C_T, L.C_D, L.C_W, L.C_H
    out = [mk.box((0.0, 0.0, L.C_BT / 2), (d, w, L.C_BT), c),  # base
           mk.box((-d / 2 + t / 2, 0.0, h / 2), (t, w, h), c),  # front wall
           mk.box((d / 2 - t / 2, 0.0, h / 2), (t, w, h), c)]  # back wall
    for s in (-1, 1):
        out.append(mk.box((0.0, s * (w / 2 - t / 2), h / 2), (L.INNER_D, t, h), c))  # side walls
        out.append(mk.box((0.0, s * (L.POCKET / 2 + L.DIV_T / 2), L.C_BT + L.DIV_H / 2),
                          (L.INNER_D, L.DIV_T, L.DIV_H), c))  # dividers
    for s in (-1, 1):
        y = s * L.HK_Y
        out.append(mk.box((d / 2 - t / 2, y, h + L.HK_TAB / 2), (t, L.HK_W, L.HK_TAB), c))  # near leg: the wall's tab
        bx0, bx1 = d / 2 - t, L.REAR_OUTER
        out.append(mk.box(((bx0 + bx1) / 2, y, L.U + L.HK_BR_T / 2), (bx1 - bx0, L.HK_W, L.HK_BR_T), c))  # bridge
        out.append(mk.box((L.REAR_INNER + L.HK_LEG_T / 2, y, L.U - L.HK_LEG_H / 2),
                          (L.HK_LEG_T, L.HK_W, L.HK_LEG_H), c))  # rear leg
        # the lead-in: the leg's tip bent outward and down
        fx = L.REAR_OUTER - L.HK_LEG_T / 2 + L.FLARE_DX / 2
        fz = L.U - L.HK_LEG_H - L.FLARE_DZ / 2 + 0.001
        out.append(mk.box((fx, y, fz), (L.HK_FLARE_L, L.HK_W, L.HK_FLARE_T), c,
                          ypr=(0.0, math.radians(L.HK_FLARE_DEG), 0.0)))
    return out


def fork_shapes():
    """A small fork standing tines down, in its own frame (origin at its middle, z from -60 to +60 mm)."""
    c, t = L.FORK_STEEL, L.FORK_T
    hz = L.FORK_L / 2
    out = []
    for dy in (-0.0065, 0.0, 0.0065):  # three 3 mm tines, 20 mm long
        out.append(mk.box((0.0, dy, -hz + 0.010), (t, 0.003, 0.020), c))
    out.append(mk.box((0.0, 0.0, -hz + 0.0275), (t, L.FORK_HEAD_W, 0.015), c))  # head
    out.append(mk.box((0.0, 0.0, -hz + 0.045), (t, 0.010, 0.020), c))  # neck
    handle = L.FORK_L - 0.055
    out.append(mk.box((0.0, 0.0, -hz + 0.055 + handle / 2), (t, 0.012, handle), c))  # handle
    return out


def wall_shapes():
    """The rack's side wall in its own frame (origin on the wall's tangent plane x X_PLANE, centre line y R_Y)."""
    c = L.RACK_GREY
    out = [mk.cylinder((L.R_X - L.X_PLANE, 0.0, L.RAIL_Z), L.RAIL_R, L.RAIL_L, "Y", c)]  # the top rail
    for z in L.WIRE_ZS:
        out.append(mk.cylinder((L.WIRE_R, 0.0, z), L.WIRE_R, L.RAIL_L, "Y", c))  # lower wires
    for s in (-1, 1):
        out.append(mk.cylinder((L.WIRE_R, s * L.POST_DY, L.POST_H / 2), L.WIRE_R, L.POST_H, "Z", c))  # end posts
    return out


def tray_shapes():
    return [mk.box((L.TRAY_LX / 2, 0.0, L.TRAY_H / 2), (L.TRAY_LX, L.TRAY_LY, L.TRAY_H), L.TRAY_GREY)]


def plate_shapes():
    z = L.TRAY_H + L.PLATE_R
    return [mk.cylinder((0.0, s * L.PLATE_DY, z), L.PLATE_R, L.PLATE_T, "Y", L.PLATE_WHITE) for s in (-1, 1)]


WALL = mk.assembly("wall", origin=(L.X_PLANE, L.R_Y, L.TOP), shapes=wall_shapes(), color=L.RACK_GREY,
                   friction=L.RACK_FRICTION, tight_fit=True)
TRAY = mk.assembly("tray", origin=(L.TRAY_X0, L.R_Y, L.TOP), shapes=tray_shapes(), color=L.TRAY_GREY,
                   friction=L.RACK_FRICTION)
PLATES = mk.assembly("plates", origin=(L.PLATE_X, L.R_Y, L.TOP), shapes=plate_shapes(), color=L.PLATE_WHITE,
                     friction=L.RACK_FRICTION)
CADDY_PART = mk.assembly(CADDY, origin=(L.START_X, L.START_Y, L.TOP + L.START_Z), mass=L.CADDY_MASS,
                         shapes=caddy_shapes(), color=L.CADDY_BLUE, friction=L.CADDY_FRICTION, tight_fit=True)
FORK_PARTS = [mk.assembly(n, origin=(L.START_X, L.START_Y + L.POCKET_Y[i], L.TOP + L.START_Z + L.FORK_Z0),
                          mass=L.FORK_MASS, shapes=fork_shapes(), color=L.FORK_STEEL, friction=L.FORK_FRICTION,
                          tight_fit=True) for i, n in enumerate(FORKS)]
OBJECTS = [WALL, TRAY, PLATES, CADDY_PART, *FORK_PARTS]
SPECS = {CADDY: CADDY_PART, **{p.name: p for p in FORK_PARTS}}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped bounding box and drops it straight down; it knows nothing of the hooks. Offline
# only (never in the simulator), this model gives the scorer: the caddy carried rigidly by a gripper closed on its two
# side walls, set down where it was let go: hung (on the wall, in its rest pose) if its hooks are over the rail with the
# rail between the legs and the legs below the rail's top, else dropped to the table where it was. The forks ride in it.
GRIP_MAX_GAP = 0.078  # the fingers closed on the 76 mm caddy (commanded 68 mm)
LET_GO_GAP = 0.080
SEAT_PLAY = 0.0005
HOOK_DEPTH = L.HK_LEG_H - 0.004  # the legs at least 4 mm down over the rail's top: engaged


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.origin = np.array([L.START_X, L.START_Y, L.START_Z])
        self.held: tuple | None = None  # (side, caddy origin - TCP)

    def _over_rail(self) -> bool:
        """The hooks straddle the rail in plan: the rail between the legs, the hooks inside the rail's ends."""
        x, y, _ = (float(v) for v in self.origin)
        return bool(L.QX_MIN - SEAT_PLAY <= L.R_X - x <= L.QX_MAX + SEAT_PLAY
                    and abs(y - L.R_Y) <= L.RAIL_L / 2 - L.HK_W - L.HK_Y)

    def _hooks_engage(self) -> bool:
        over = float(self.origin[2]) + L.U - L.RAIL_TOP
        return bool(self._over_rail() and -L.SEAT_Z_LO <= over <= HOOK_DEPTH)

    def _drop(self) -> None:
        if self._hooks_engage():
            self.origin = np.array([L.HUNG_X, float(self.origin[1]), L.HUNG_Z])
        else:
            self.origin = np.array([float(self.origin[0]), float(self.origin[1]), 0.0])

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        sides = dict(zip(rigstate.ARMS, read))
        top = np.array([0.0, 0.0, L.TOP])
        if self.held is not None:
            side, off = self.held
            tcp, _, opening = sides[side]
            if opening * rigstate.FULL_GAP > LET_GO_GAP:
                self.held = None
                self._drop()
            else:
                self.origin = (np.asarray(tcp, float) - top) + off
                if self._over_rail() and float(self.origin[2]) < L.HUNG_Z:
                    self.origin[2] = L.HUNG_Z  # the bridges bear on the rail: the hand cannot push it lower
        if self.held is None:
            for side, (tcp, _, opening) in sides.items():
                gap = opening * rigstate.FULL_GAP
                t = np.asarray(tcp, float) - top
                d = t - (self.origin + np.array([L.GRIP_X, 0.0, GRIP_DZ]))
                if gap <= GRIP_MAX_GAP and abs(float(d[0])) <= 0.012 and abs(float(d[1])) <= 0.012 \
                        and abs(float(d[2])) <= 0.012:
                    self.held = (side, self.origin - t)
                    break

    def states(self) -> dict:
        q0 = (0.0, 0.0, 0.0, 1.0)
        x, y, z = (float(v) for v in self.origin)
        out = {CADDY: mk.state_from_poses(CADDY_PART, (x, y, L.TOP + z, *q0), source="offline")}
        for i, n in enumerate(FORKS):
            out[n] = mk.state_from_poses(SPECS[n], (x, y + L.POCKET_Y[i], L.TOP + z + L.FORK_Z0, *q0),
                                         source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(CADDY), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only, every 2nd step: [step, base mm, rail qx mm, bridge gap mm, yaw, tilt]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: every key part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Both hooks seated on the rail, the caddy square, all three forks in it, both grippers 30 mm clear and the caddy
    still for 1 s, no latched fall; held 1 s. This runs first every control step, so it also feeds the tracker (and,
    without a simulator, the offline stand-in)."""
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
    row["source"] = getattr(parts.get(CADDY), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["base_mm"], d["rail_qx_mm"]["1"], d["bridge_gap_mm"]["1"], d["yaw_deg"],
                      d["tilt_deg"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the caddy or a fork fell off the table, or unless the caddy is let go and still; 1.0 both hooks seated on
    the rail, the caddy square and the three forks in it; 0.5 the same with a fork out; 0.25 one hook seated, or both
    with the caddy crooked; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the caddy).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm: 9 mm each side of the 76 mm caddy
GRIP_F = -(L.C_W - 0.008) / 2  # 68 mm commanded across the side walls
YAW = 0.0  # both pads close along y, across the caddy's width
GRIP_TIPS_DOWN = 0.015  # pad tips 15 mm below the rim: the housing stays 25 mm above the forks' tops
GRIP_DZ = L.C_H - GRIP_TIPS_DOWN + PAD_TIP  # the TCP's height over the caddy's base: 98 mm
GRIP_GX = L.START_X + L.GRIP_X  # the TCP's x at the caddy's start
CARRY_Z = 0.076  # the caddy raised this far (its base's underside): the flared tips 24 mm over the rail's top (the
# loaded arm sags 3 to 7 mm)
DEMO_X = 0.312  # the caddy's x when lowered: the rail's axis 18 mm ahead of it, 4 mm clear of each leg
PRESS = 0.004  # lowered 4 mm past the hung height, so the bridges bear on the rail before the fingers open
HOVER = 0.180
PARK = (0.18, -0.32, 0.18)


def _w(d, x, y, z, finger, label, yaw=YAW):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def plans(variant: str = "") -> dict:
    """The right arm's plan for the demo or a counterfactual variant: grip the caddy by its two side walls, lift it
    over the rail, carry it, lower it over the rail with the hooks straddling it, let go, up, aside."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    x_c = DEMO_X + VARIANT_DX.get(variant, 0.0)
    gx, gy = x_c + L.GRIP_X, L.R_Y
    gz_up = GRIP_DZ + CARRY_Z
    gz_hung = GRIP_DZ + L.HUNG_Z - PRESS
    right = [
        _w(2.0, GRIP_GX, L.START_Y, HOVER, OPEN, "over the caddy"),
        _w(1.0, GRIP_GX, L.START_Y, GRIP_DZ + 0.020, OPEN, "down to just above its rim"),
        _w(1.5, GRIP_GX, L.START_Y, GRIP_DZ, OPEN, "slowly down round its side walls"),
        _w(0.5, GRIP_GX, L.START_Y, GRIP_DZ, OPEN, "settle"),
        _w(1.5, GRIP_GX, L.START_Y, GRIP_DZ, GRIP_F, "grip the caddy across its width"),
        _w(2.0, GRIP_GX, L.START_Y, gz_up, GRIP_F, "lift it clear of the rail"),
        _w(3.0, gx, gy, gz_up, GRIP_F, "carry it over the rail"),
        _w(1.0, gx, gy, gz_up, GRIP_F, "settle over the rail"),
        _w(2.5, gx, gy, gz_hung, GRIP_F, "lower the hooks over the rail"),
        _w(0.5, gx, gy, gz_hung, GRIP_F, "settle on the rail"),
        _w(1.5, gx, gy, gz_hung, OPEN, "let go"),
        _w(1.5, gx, gy, HOVER, OPEN, "straight up"),
        _w(2.0, *PARK, OPEN, "aside"),
    ]
    return {"right": right}


# Wrong ways (counterfactuals): the caddy's x when lowered, off the demo's by this much.
VARIANT_DX = {"short": -0.015, "long": 0.012}
VARIANTS = tuple(VARIANT_DX)


def demo():
    """The right arm on its own clock; the left arm holds the ready pose. DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short  lowered 15 mm too far toward the robot: the rear legs land on top of the rail
      long   lowered 12 mm too far toward the rack: the back wall's tabs land on top of the rail"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
