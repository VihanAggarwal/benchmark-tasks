"""On the station table, real size: a stainless mixing bowl (220 mm across its rolled rim, 105 mm deep) on a non-slip
mat, holding about half a litre of strained stock (a solid disc in the model, 32 mm up the bowl); a stock pot and a
backsplash wall behind them; and, 25 cm to the bowl's left, a fine-mesh strainer lying mesh down, long axis along y,
handle toward the left arm: a 130 mm ring 57 mm tall, a 130 mm folded handle, and a hook (a strap bent over, 30 mm
gap) on the other side. The strainer is picked up by its handle and hung across the bowl: the hook over the right
rim, the handle on the left rim, the mesh hanging in the bowl 26 mm clear of the stock. Then let go and take the hand
away.

The bowl, the strainer, the mat, the pot and the wall are mechkit compounds (mechkit.py, next to this file; no joints,
no rules). The arms are read through rigstate.py (the let-go check). The scorer reads poses only.

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
from scoring import BOWL, STRAINER  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "Hang the fine-mesh strainer across the bowl: its hook over the right rim and its handle on the left rim, the "
    "mesh hanging in the bowl clear of the stock. Then let go."
)


# -- the parts --------------------------------------------------------------------------------------------------------
def bowl_shapes():
    """The bowl in its own frame (origin at the centre of the foot's underside, z up): a floor, 24 flared wall slabs
    (inner radius 70 mm at the floor, 100 mm at the rim), the rolled bead as 24 boxes (radius 102 to 110 mm, 8 mm
    tall), and the stock as a solid disc 30 mm deep over the floor."""
    th = math.atan(L.WALL_SLOPE)
    zm = (L.BOWL_FLOOR + L.BOWL_H) / 2
    r_in = L.R_FLOOR_IN + L.WALL_SLOPE * (zm - L.BOWL_FLOOR)
    rc = r_in + L.BOWL_WALL / 2 / math.cos(th)
    height = (L.BOWL_H - L.BOWL_FLOOR) / math.cos(th)
    seg = 2 * (L.BEAD_OUT - 0.008) * math.tan(math.pi / L.WALL_SEGMENTS) + 0.003
    out = [mk.cylinder((0.0, 0.0, L.BOWL_FLOOR / 2), L.R_FLOOR_IN + L.BOWL_WALL, L.BOWL_FLOOR, color=L.BOWL_STEEL)]
    for k in range(L.WALL_SEGMENTS):
        a = 2 * math.pi * k / L.WALL_SEGMENTS
        out.append(mk.box((rc * math.cos(a), rc * math.sin(a), zm), (L.BOWL_WALL, seg, height), L.BOWL_STEEL,
                          ypr=(a, th, 0.0)))
    bead_c = (L.R_RIM_IN + L.BOWL_WALL + L.BEAD_OUT) / 2
    bead_w = L.BEAD_OUT - (L.R_RIM_IN + L.BOWL_WALL)
    bseg = 2 * L.BEAD_OUT * math.tan(math.pi / L.BEAD_SEGMENTS) + 0.003
    for k in range(L.BEAD_SEGMENTS):
        a = 2 * math.pi * k / L.BEAD_SEGMENTS
        out.append(mk.box((bead_c * math.cos(a), bead_c * math.sin(a), L.BOWL_H - L.BEAD_H / 2),
                          (bead_w, bseg, L.BEAD_H), L.BOWL_STEEL, ypr=(a, 0.0, 0.0)))
    out.append(mk.cylinder((0.0, 0.0, L.BOWL_FLOOR + L.STOCK_DEPTH / 2), L.STOCK_R, L.STOCK_DEPTH,
                           color=L.STOCK_COLOR))
    return out


def strainer_shapes():
    """The strainer in its own frame (origin at the ring's centre on the rest plane, local x toward the handle): the
    mesh as a 2 mm solid floor, the ring's wall as 16 slabs, the hook (strap, outer leg, inner leg) and the folded
    handle (a web over two flanges whose bottoms are in the rest plane)."""
    s = L.STRAINER_STEEL
    out = [mk.cylinder((0.0, 0.0, -L.RING_DEPTH + L.MESH_T / 2), L.RING_R, L.MESH_T, color=s)]
    wall_h = L.RING_DEPTH + L.RING_TOP
    seg = 2 * L.RING_R * math.tan(math.pi / L.RING_SEGMENTS) + 0.003
    rc = L.RING_R - L.RING_WALL / 2
    for k in range(L.RING_SEGMENTS):
        a = 2 * math.pi * k / L.RING_SEGMENTS
        out.append(mk.box((rc * math.cos(a), rc * math.sin(a), (L.RING_TOP - L.RING_DEPTH) / 2),
                          (L.RING_WALL, seg, wall_h), s, ypr=(a, 0.0, 0.0)))
    ring_edge = -(L.RING_R - L.RING_WALL)
    strap_len = ring_edge - L.HK_END
    out.append(mk.box(((L.HK_END + ring_edge) / 2, 0.0, L.STRAP_T / 2), (strap_len, L.HK_W, L.STRAP_T), s))
    out.append(mk.box((L.HK_END + L.STRAP_T / 2, 0.0, (L.STRAP_T - L.HK_LEG_OUT) / 2),
                      (L.STRAP_T, L.HK_W, L.HK_LEG_OUT + L.STRAP_T), s))
    out.append(mk.box((L.HK_IN - L.STRAP_T / 2, 0.0, (L.STRAP_T - L.HK_LEG_IN) / 2),
                      (L.STRAP_T, L.HK_W, L.HK_LEG_IN + L.STRAP_T), s))
    hx0 = L.RING_R - L.RING_WALL
    hlen = L.HD_X1 - hx0
    hc = (hx0 + L.HD_X1) / 2
    out.append(mk.box((hc, 0.0, L.HD_FLANGE_H - L.STRAP_T / 2), (hlen, L.HD_W, L.STRAP_T), s))
    for sg in (-1, 1):
        out.append(mk.box((hc, sg * (L.HD_W / 2 - L.HD_FLANGE_T / 2), L.HD_FLANGE_H / 2),
                          (hlen, L.HD_FLANGE_T, L.HD_FLANGE_H), s))
    return out


MAT = mk.assembly("mat", origin=(L.MAT_X, L.MAT_Y, L.TOP),
                  shapes=[mk.box((0.0, 0.0, L.MAT_SIZE[2] / 2), L.MAT_SIZE, L.MAT_COLOR)], color=L.MAT_COLOR,
                  friction=L.MAT_FRICTION, tight_fit=True)
WALL = mk.assembly("wall", origin=(L.WALL_X + L.WALL_SIZE[0] / 2, 0.0, L.TOP),
                   shapes=[mk.box((0.0, 0.0, L.WALL_SIZE[2] / 2), L.WALL_SIZE, L.WALL_COLOR)], color=L.WALL_COLOR,
                   friction=L.SCENE_FRICTION)
POT = mk.assembly("pot", origin=(L.POT_X, L.POT_Y, L.TOP),
                  shapes=[mk.cylinder((0.0, 0.0, L.POT_H / 2), L.POT_R, L.POT_H, color=L.POT_COLOR)],
                  color=L.POT_COLOR, friction=L.SCENE_FRICTION)
BOWL_PART = mk.assembly(BOWL, origin=(L.BOWL_X, L.BOWL_Y, L.BOWL_Z0), mass=L.BOWL_MASS, shapes=bowl_shapes(),
                        color=L.BOWL_STEEL, friction=L.BOWL_FRICTION, tight_fit=True)
STRAINER_PART = mk.assembly(STRAINER, origin=(L.ST_X, L.ST_Y, L.ST_Z), yaw=L.ST_YAW, mass=L.STRAINER_MASS,
                            shapes=strainer_shapes(), color=L.STRAINER_STEEL, friction=L.STRAINER_FRICTION,
                            tight_fit=True)
OBJECTS = [MAT, WALL, POT, BOWL_PART, STRAINER_PART]
SPECS = {STRAINER: STRAINER_PART, BOWL: BOWL_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part as its bounding box and drops it onto the highest thing below; it does not
# hang a hook on a rim. Offline only (never in the simulator), this model gives the scorer the strainer's pose: carried
# rigidly by a closed gripper whose TCP is on its handle; on being let go it comes to rest (1) hooked, when its axis is
# within 40 mm of the bowl's and the hook's gap straddles the near bead (the window below); (2) resting on an inner or
# outer leg (raised 12 or 20 mm at the hook end, pitched), when it is over the bowl but outside that window; (3) in the
# stock, when the ring is over the bowl more than 40 mm off its axis; (4) on the table otherwise. The bowl stays put.
CATCH_LO_FLARE = 0.0065  # the inner leg's foot (12 mm down the flared wall) meets the wall at s = -6.5 mm
CATCH_HI = (-L.HK_OUT) - L.BEAD_OUT  # 0.010: the outer leg's face reaches the bead's outer face at s = +10 mm
GRIP_LOCAL = (L.GRIP_X, 0.0, 0.005)  # where the pads close on the handle (local x, y, z)
GRIP_X_TOL = 0.025


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.origin = np.array([L.ST_X, L.ST_Y, L.ST_Z])
        self.pitch = 0.0
        self.grip: np.ndarray | None = None  # origin minus TCP while carried

    def _quat(self):
        return mk.quat_from_ypr(L.ST_YAW, self.pitch, 0.0)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        r = mk.quat_to_matrix(self._quat())
        handle = self.origin + r @ np.array(GRIP_LOCAL)
        held_by = None
        for tcp, tips, opening in read:
            near = (abs(float(tcp[0]) - handle[0]) < GRIP_X_TOL and abs(float(tcp[1]) - handle[1]) < GRIP_X_TOL
                    and -0.005 <= float(tips[2]) - (self.origin[2] + 0.002) <= 0.02)
            if opening < 0.3 and (near or self.grip is not None):
                held_by = tcp
                break
        if held_by is not None:
            if self.grip is None:
                self.grip = self.origin - held_by
            self.origin = held_by + self.grip
            self.pitch = 0.0
            return
        if self.grip is not None:  # just let go: come to rest
            self.grip = None
            self._settle()

    def _settle(self) -> None:
        dx = self.origin[0] - L.BOWL_X
        s = self.origin[1] - L.BOWL_Y  # along the strainer's axis (the handle toward +y)
        bead = L.BEAD_TOP_W
        self.pitch = 0.0
        if math.hypot(dx, s) > 0.075:  # not over the bowl: dropped onto the table
            if self.origin[2] > L.ST_Z + 0.005:
                self.origin[2] = L.ST_Z
            return
        if abs(dx) > 0.040:  # the ring over the bowl without a rim to rest on: it sinks onto the stock
            self.origin[2] = L.BOWL_Z0 + L.STOCK_TOP_B + L.RING_DEPTH
        elif -CATCH_LO_FLARE <= s <= CATCH_HI:  # the hook's legs straddle the near bead, the flanges on the far one
            self.origin[2] = bead
        elif s < -CATCH_LO_FLARE:  # the inner leg's foot lands on the bead: the hook end is raised 12 mm
            self.origin[2] = bead + 0.006
            self.pitch = math.atan(L.HK_LEG_IN / (L.RIM_R * 2))
        else:  # the outer leg's foot lands on the bead: raised 20 mm
            self.origin[2] = bead + 0.010
            self.pitch = math.atan(L.HK_LEG_OUT / (L.RIM_R * 2))

    def states(self) -> dict:
        st = mk.state_from_poses(STRAINER_PART, (*self.origin, *self._quat()), source="offline")
        o, q = BOWL_PART.spawn_pose()
        return {STRAINER: st, BOWL: mk.state_from_poses(BOWL_PART, (*o, *q), source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(STRAINER), "source", "sim") != "sim"


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
    """The hook over the right rim, the handle on the left rim, the ring in the bowl with the mesh clear of the stock,
    the strainer level, both grippers 30 mm clear and both parts still for 1 s, no latch; held 1 s. This runs first
    every control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
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
    row["source"] = getattr(parts.get(STRAINER), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the bowl was moved 15 mm, tipped 10 degrees or lifted, or a part fell off the counter, or unless both parts
    are let go and still; 1.0 hooked on the right rim, the handle on the left rim, the ring in the bowl, the mesh 15 mm
    or more above the stock, level; 0.5 across the rims with the mesh clear but not hooked (or hooked off line or on
    another part of the rim); 0 otherwise (a mesh down in the stock included). The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points (z above the
# table top); the pad tips are 13 mm below. Finger targets: the joint value is minus half the commanded gap (the grip
# 10 mm narrower than the 18 mm handle).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
OPEN_H = -0.021  # 42 mm round the 18 mm handle: 12 mm each side
GRIP_H = -0.001  # 2 mm commanded across the 18 mm handle: 8 mm short a pad, about 40 N a pad (the first GPU run's 8 mm
# command, about 25 N, let the strainer turn in the pinch: it holds no torque about its closing axis, only friction)
CLOSED = 0.0
YAW_L = -math.pi / 2  # the left arm closes along x

GRIP_LOCAL_X = L.GRIP_X  # the pads close on the handle 82 mm from the ring's centre, next to the ring
START_GRIP_Y = L.ST_Y + GRIP_LOCAL_X  # 0.334
TARGET_S = L.HOOK_AIM_S
CATCH_MID = (-CATCH_LO_FLARE + CATCH_HI) / 2  # the middle of the hooking window along the axis: +1.75 mm
TIPS_UP = 0.0  # pad tips level with the flanges' bottoms: all of the flanges' 10 mm (the wider part of the pad)
GRIP_TCP = L.RING_DEPTH + TIPS_UP + PAD_TIP  # 0.060: at the start
CARRY_LIP = L.BEAD_TOP_W - L.TOP + L.RING_DEPTH + 0.014  # the rest plane 170 mm up: the mesh 14 mm over the bead
CARRY_TCP = CARRY_LIP + TIPS_UP + PAD_TIP
HOVER_LIP = L.BEAD_TOP_W - L.TOP + 0.012  # 12 mm above the bead: the hook's outer leg is 8 mm below the bead's top
# The carry stays high only until the ring's edge has crossed the far (left) bead, then comes down across the bowl, so
# the hand is never at full carry height over the bowl's middle (the left wrist's joint 4 has little room there).
PASS_S = 0.033  # the ring's centre 33 mm toward the handle from its final place: its edge at the far rim's inner edge
INSIDE_S = 0.015  # 15 mm: the ring's edge 20 mm inside the far bead, the hook's outer leg just inside the near bead
INSIDE_LIP = L.BEAD_TOP_W - L.TOP + 0.045  # 45 mm above the bead: the hook's outer leg about 18 mm over the near bead's top
# (the second GPU run, at 28 mm, put the leg 3 mm below the bead's top after the arm's 6.6 mm sag: it jammed against the
# bowl's inner wall 3 mm short of the bead and the strainer hung with its hook 16 mm up, 24 mm toward the handle)
SET_LIP = L.BEAD_TOP_W - L.TOP - 0.0005  # commanded half a millimetre below the rest height; the sag does the rest
HOVER_H = 0.140
ASIDE = (0.30, 0.385, 0.150)
UP_TCP = 0.165  # lifted 33 mm clear of the handle after the let-go (low enough to keep joint 4 off its stop)


def _w(d, x, y, z, finger, label, yaw=YAW_L):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


# Wrong ways (counterfactuals): where the ring's centre is let go along the axis from the bowl's axis, and how low.
VARIANT_S = {"inner_on_rim": -0.020, "outer_on_rim": 0.025}
VARIANT_DX = {"off_centre": 0.025}
VARIANTS = tuple(VARIANT_S) + tuple(VARIANT_DX)


def plans(variant: str = "") -> dict:
    """The left arm's plan for the demo or a counterfactual variant: down round the handle, grip it, lift it straight up
    to carry height (mesh 20 mm over the bead), carry it across to the bowl, lower it so the hook's legs straddle the near
    bead and the handle's flanges land on the far bead, let go, up and aside. The right arm holds its ready pose."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    s = VARIANT_S.get(variant, TARGET_S)
    x = L.ST_X
    x1 = x + VARIANT_DX.get(variant, 0.0)  # where the strainer is set down
    y0 = START_GRIP_Y
    y1 = L.BOWL_Y + s + GRIP_LOCAL_X
    set_tcp = SET_LIP + TIPS_UP + PAD_TIP
    plan = [
        _w(2.0, x, y0, HOVER_H, OPEN_H, "over the handle"),
        _w(1.0, x, y0, GRIP_TCP + 0.030, OPEN_H, "down to just above it"),
        _w(1.5, x, y0, GRIP_TCP, OPEN_H, "slowly down round the handle"),
        _w(0.5, x, y0, GRIP_TCP, OPEN_H, "settle"),
        _w(1.5, x, y0, GRIP_TCP, GRIP_H, "grip the handle across its width"),
        _w(0.5, x, y0, GRIP_TCP, GRIP_H, "settle the grip"),
        _w(3.0, x, y0, CARRY_TCP, GRIP_H, "lift it straight up, the mesh clear of the bowl's rim"),
        _w(3.25, x1, y1 + PASS_S, CARRY_TCP, GRIP_H, "carry it across: the ring's edge over the far bead"),
        _w(0.5, x1, y1 + PASS_S, CARRY_TCP, GRIP_H, "settle"),
        _w(1.0, x1, y1 + INSIDE_S, INSIDE_LIP + TIPS_UP + PAD_TIP, GRIP_H,
           "down and across: the ring inside the far rim, the hook's legs over the near bead"),
        _w(0.5, x1, y1, INSIDE_LIP + TIPS_UP + PAD_TIP, GRIP_H, "across: the hook's outer leg past the near bead"),
        _w(0.75, x1, y1, HOVER_LIP + TIPS_UP + PAD_TIP, GRIP_H, "down until the hook is 12 mm over the near bead"),
    ]
    plan += [
        _w(2.0, x1, y1, set_tcp, GRIP_H, "slowly down: the hook's legs straddle the bead, the handle on the far rim"),
        _w(0.5, x1, y1, set_tcp, GRIP_H, "settle"),
        _w(1.5, x1, y1, set_tcp, OPEN_H, "let go"),
        _w(1.0, x1, y1, UP_TCP, OPEN_H, "straight up"),
    ]
    plan.append(_w(1.5, ASIDE[0], ASIDE[1], ASIDE[2], OPEN, "aside, out of the way"))
    return {"left": plan}


def demo():
    """The left arm alone (the right arm holds its ready pose). DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      inner_on_rim  the ring let go 20 mm short of the hooking window: the inner leg lands on the bead (0.5)
      outer_on_rim  25 mm past it: the outer leg lands on the bead (0.5)
      off_centre    the whole carry 25 mm to the side: hooked, but the axis misses the bowl's by 25 mm (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
