"""On the station table, real size: a pick cart's bed (a 6 mm steel deck, a tote locator frame round each of its two
bays, a rear frame with a push handle) and two loose stacking totes (400 x 300 x 64 mm, polypropylene, 0.70 kg each with
the picked goods in them), one in each bay, 60 mm apart end to end. The tote in the +y bay is the picked one. Each tote
has a hand hold (a 110 mm slot under a 10 mm grip bar) in each of its two end walls.

Goal: the picked tote lifted out of its bay by its two end hand holds (one hand each), carried across, and set on the
tote in the -y bay with its 386 x 286 mm foot nested inside the lower tote's walls (4 mm clear all round) and its ledge
on the lower walls' top: seated 55 mm above the lower tote's foot, centred to 4 mm, square to 2 degrees, level to
1 degree, the lower tote not disturbed, both hands 30 mm clear and both totes still for 1 s. A tote set down more than
10 mm off centre does not drop in: it rides on the rim, 9 mm higher.

The totes and the cart are mechkit compounds (mechkit.py, next to this file). The arms are read through rigstate.py
(the let-go check). No software rule moves anything: every effect is the bodies' own contact.

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
from scoring import BASE, PICK  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Stack the tote in the left bay onto the tote in the right bay, seated and square, and let go."


# -- the parts --------------------------------------------------------------------------------------------------------
def _box(cx, cy, cz, sx, sy, sz, color, **kw):
    return mk.box((cx, cy, cz), (sx, sy, sz), color, **kw)


def _lead_in(sign_y: bool, sign: int):
    """One 45 degree lead-in plate on the foot's underside: from the floor's edge (flush with the floor's underside)
    up to the foot wall's outer face 6 mm higher. ``sign_y`` True: along an end (normal y), else along a side."""
    t = 0.003
    w = L.FOOT_CH * math.sqrt(2.0)
    r_in = (L.FLOOR_L if sign_y else L.FLOOR_W) / 2
    r_out = (L.FOOT_L if sign_y else L.FOOT_W) / 2
    mid_r, mid_z = (r_in + r_out) / 2, L.FOOT_CH / 2
    r_c = mid_r - t / 2 / math.sqrt(2.0)  # the plate's outer face runs through (r_in, 0) and (r_out, FOOT_CH)
    z_c = mid_z + t / 2 / math.sqrt(2.0)
    if sign_y:
        return _box(0.0, sign * r_c, z_c, L.FOOT_W - 0.006, w, t, L.TOTE_BLUE, ypr=(0.0, 0.0, sign * math.pi / 4))
    return _box(sign * r_c, 0.0, z_c, w, L.FOOT_L - 0.014, t, L.TOTE_BLUE, ypr=(0.0, -sign * math.pi / 4, 0.0))


def tote_shapes():
    """One stacking tote in its own frame (origin at the centre of the foot's underside): the floor and its lead-in
    plates, the foot wall, the ledge ring (underside 9 mm up: what the stack seats on), the main walls to 74 mm, the
    end walls with their hand-hold slot and grip bar, and the carton of goods on the floor."""
    c = L.TOTE_BLUE
    t = L.WALL
    out = [_box(0.0, 0.0, L.FLOOR_T / 2, L.FLOOR_W, L.FLOOR_L, L.FLOOR_T, c)]
    out += [_lead_in(True, s) for s in (-1, 1)] + [_lead_in(False, s) for s in (-1, 1)]
    # the foot wall: 386 x 286 mm outside, from 6 to 12 mm
    fz, fh = (L.FOOT_CH + L.FOOT_H) / 2, L.FOOT_H - L.FOOT_CH
    for s in (-1, 1):
        out.append(_box(0.0, s * (L.FOOT_L / 2 - t / 2), fz, L.FOOT_W, t, fh, c))
        out.append(_box(s * (L.FOOT_W / 2 - t / 2), 0.0, fz, t, L.FOOT_L - 2 * t, fh, c))
    # the ledge: a flat ring from the foot wall out to the outer face, 9 to 12 mm
    lz, lh = (L.LEDGE_Z0 + L.LEDGE_Z1) / 2, L.LEDGE_Z1 - L.LEDGE_Z0
    for s in (-1, 1):
        out.append(_box(0.0, s * (L.TOTE_L / 2 - 0.005), lz, L.TOTE_W, 0.010, lh, c))
        out.append(_box(s * (L.TOTE_W / 2 - 0.005), 0.0, lz, 0.010, L.TOTE_L - 0.020, lh, c))
    # the main walls, 9 to 74 mm
    wz, wh = (L.LEDGE_Z0 + L.TOTE_H) / 2, L.TOTE_H - L.LEDGE_Z0
    for s in (-1, 1):
        out.append(_box(s * (L.TOTE_W / 2 - t / 2), 0.0, wz, t, L.TOTE_L, wh, c))
    # the end walls (normal y): solid to 24 mm, the slot 24 to 40 mm, the grip bar 40 to 52 mm, the rim wall to 74 mm
    for s in (-1, 1):
        yw = s * (L.TOTE_L / 2 - t / 2)
        out.append(_box(0.0, yw, (L.LEDGE_Z0 + L.HH_SLOT_Z0) / 2, L.TOTE_W, t, L.HH_SLOT_Z0 - L.LEDGE_Z0, c))
        px0, px1 = L.HH_SLOT_W / 2, L.TOTE_W / 2
        for sx in (-1, 1):
            out.append(_box(sx * (px0 + px1) / 2, yw, (L.HH_SLOT_Z0 + L.HH_SLOT_Z1) / 2, px1 - px0, t,
                            L.HH_SLOT_Z1 - L.HH_SLOT_Z0, c))
        out.append(_box(0.0, s * L.BAR_Y, L.GRIP_Z, L.TOTE_W, L.BAR_T, L.BAR_Z1 - L.BAR_Z0, c))
        out.append(_box(0.0, yw, (L.BAR_Z1 + L.TOTE_H) / 2, L.TOTE_W, t, L.TOTE_H - L.BAR_Z1, c))
    cx, cy, cz = L.CARTON
    out.append(_box(0.0, 0.0, L.FLOOR_T + cz / 2, cx, cy, cz, L.CARTON_TAN))
    return out


def cart_shapes():
    """The cart's bed in the frame at (BAY_X, 0, table top): the deck, a locator frame round each bay (4 mm clear of
    the foot), the rear uprights and the push handle."""
    g = L.CART_GREY
    dx = (L.DECK_X0 + L.DECK_X1) / 2 - L.BAY_X
    out = [_box(dx, 0.0, L.DECK_T / 2, L.DECK_X1 - L.DECK_X0, L.DECK_Y, L.DECK_T, g)]
    rz = L.DECK_T + L.RAIL_H / 2
    for yc in (L.PICK_Y, L.BASE_Y):
        for s in (-1, 1):
            out.append(_box(0.0, yc + s * (L.FOOT_L / 2 + L.CLEAR + L.RAIL / 2), rz,
                            L.FOOT_W + 2 * L.CLEAR + 2 * L.RAIL, L.RAIL, L.RAIL_H, g))
            out.append(_box(s * (L.FOOT_W / 2 + L.CLEAR + L.RAIL / 2), yc, rz,
                            L.RAIL, L.FOOT_L + 2 * L.CLEAR, L.RAIL_H, g))
    px = L.POST_X - L.BAY_X
    for s in (-1, 1):
        out.append(_box(px, s * L.POST_Y, L.DECK_T + L.POST_H / 2, L.POST, L.POST, L.POST_H, g))
    out.append(_box(px, 0.0, L.DECK_T + L.POST_H - L.POST / 2, L.POST, 2 * L.POST_Y + L.POST, L.POST, g))
    return out


CART = mk.assembly("cart", origin=(L.BAY_X, 0.0, L.TOP), shapes=cart_shapes(), color=L.CART_GREY,
                   friction=L.CART_FRICTION, tight_fit=True)
BASE_PART = mk.assembly(BASE, origin=(L.BASE_ORIGIN[0], L.BASE_ORIGIN[1], L.TOP + L.BASE_ORIGIN[2]),
                        mass=L.TOTE_MASS, shapes=tote_shapes(), color=L.TOTE_BLUE, friction=L.TOTE_FRICTION,
                        tight_fit=True)
PICK_PART = mk.assembly(PICK, origin=(L.PICK_ORIGIN[0], L.PICK_ORIGIN[1], L.TOP + L.PICK_ORIGIN[2]),
                        mass=L.TOTE_MASS, shapes=tote_shapes(), color=L.TOTE_BLUE, friction=L.TOTE_FRICTION,
                        tight_fit=True)
OBJECTS = [CART, BASE_PART, PICK_PART]
SPECS = {BASE: BASE_PART, PICK: PICK_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped box rigidly and drops it straight down onto the highest thing under it; it has
# no nest to seat in. Offline only (never in the simulator), this model gives the scorer: (1) the picked tote carried
# by BOTH hands when each is closed on its own end's grip bar (one hand does not lift it: it would hang by one end),
# at the hands' mean position; (2) on release, a foot within the lead-in's 10 mm of the lower tote's centre drops in
# and centres to the 4 mm of clearance (seated), one 10 to 13 mm off rides on the wall tops 9 mm higher, any other
# lands on the deck (or the table off it).
HOLD_XY = 0.015  # a hand counts as on its bar within this (m) of it in the plan view
HOLD_Z = 0.020
HOLD_OPEN = 0.30  # closed on the bar: the opening is 2 mm of 94
RIDE_MAX = 0.013  # the wall's outer face: further off the foot slides down beside it


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = np.array(L.PICK_ORIGIN, dtype=float)  # the picked tote's origin (table top at 0)
        self.held = False

    def _holding(self, read) -> bool:
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            bar_y = self.pos[1] + (L.BAR_Y if side == "left" else -L.BAR_Y)
            if not (opening < HOLD_OPEN and abs(float(tcp[0]) - self.pos[0]) <= HOLD_XY
                    and abs(float(tcp[1]) - bar_y) <= HOLD_XY
                    and abs(float(tcp[2]) - L.TOP - (self.pos[2] + L.GRIP_Z + 0.002)) <= HOLD_Z):
                return False
        return True

    def _land(self) -> None:
        dx, dy = self.pos[0] - L.BASE_ORIGIN[0], self.pos[1] - L.BASE_ORIGIN[1]
        low = self.pos[2]
        if low >= L.BASE_ORIGIN[2] + L.SEAT_DZ - 0.015 and L.seat_offset_ok(dx, dy):
            dx, dy = L.seated_offsets(dx, dy)
            self.pos = np.array([L.BASE_ORIGIN[0] + dx, L.BASE_ORIGIN[1] + dy, L.BASE_ORIGIN[2] + L.SEAT_DZ])
        elif low >= L.BASE_ORIGIN[2] + L.SEAT_DZ - 0.015 and abs(dx) <= RIDE_MAX and abs(dy) <= RIDE_MAX:
            self.pos = np.array([self.pos[0], self.pos[1], L.BASE_ORIGIN[2] + L.RIDE_DZ])
        else:
            on_deck = abs(self.pos[0] - L.BAY_X) <= 0.17 and abs(self.pos[1]) <= L.DECK_Y / 2
            self.pos = np.array([self.pos[0], self.pos[1], L.DECK_T if on_deck else 0.0])

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        if self._holding(read):
            self.held = True
            tcp = np.mean([r[0] for r in read], axis=0)
            self.pos = np.array([tcp[0], tcp[1], float(tcp[2]) - L.TOP - L.GRIP_Z - 0.002])
        elif self.held:
            self.held = False
            self._land()

    def states(self) -> dict:
        q0 = (0.0, 0.0, 0.0, 1.0)
        base = mk.state_from_poses(BASE_PART, (L.BASE_ORIGIN[0], L.BASE_ORIGIN[1], L.TOP + L.BASE_ORIGIN[2], *q0),
                                   source="offline")
        pick = mk.state_from_poses(PICK_PART, (self.pos[0], self.pos[1], L.TOP + self.pos[2], *q0),
                                   source="offline")
        return {BASE: base, PICK: pick}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(BASE), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, seat error, dx, dy (mm), tilt deg, base moved mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: both totes' poses and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The picked tote seated on the base tote (55 mm up, centred to 4 mm, square to 2 degrees, level to 1 degree),
    the base tote undisturbed, both grippers 30 mm clear and both totes still for 1 s, no latched fall; held 1 s. This
    runs first every control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
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
    row["source"] = getattr(parts.get(BASE), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["seat_error_mm"], d["dx_mm"], d["dy_mm"], d["tilt_deg"], d["base_moved_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if either tote fell off the deck or tipped on its side, the base tote was pushed more than 10 mm or turned
    more than 5 degrees, or unless both hands are clear and both totes still; 1.0 seated and square; 0.5 resting on the
    base tote within 20 mm of its centre; 0.25 on it otherwise; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
OPEN_G = -0.018  # 36 mm: round the 10 mm bar with 13 mm each side, and the outer pad stays clear of the other tote
GRIP_F = -(L.BAR_T - 0.008) / 2  # 2 mm commanded across the 10 mm bar: about 20 N a finger
CLOSED = 0.0
YAW = 0.0  # both close along y, across the end walls

X = L.BAY_X
GRIP_Z_PICK = L.GRIP_TCP_PICK  # 54 mm: the bar's centre height with the tote on the deck, the pad tips 13 mm below
GRIP_Z_SEAT = L.SEAT_Z + L.GRIP_Z + 0.002  # 109 mm: the same with the tote seated
HOVER = 0.120
CARRY = 0.145  # the foot's underside 17 mm over the lower walls' top (commanded; the arms sag under the load)
RETREAT = 0.175  # after letting go: the pad tips (13 mm below the TCP, sag under 1 mm) 17 mm over the upper tote's
# rim (seated rim 145 mm above the table; with the 74 mm tote the 145 mm carry height left the tips 13 mm below it,
# so the pads dragged on the end wall and the right hand never got aside: GPU run 20261116T221712)
# slide 90 mm toward the arm bases on the way up: the left hand's joint 4 has 6.7 degrees there (0.6 in line)
RETREAT_DX = -0.09
ABOVE_SEAT = 0.136  # still 8 mm above the walls' top
PRESS = 0.004  # lowered 4 mm past seated, so the ledge rests its full weight on the walls before the hands open
CARRY_S = 5.0
PARK_Y = {"left": +0.075, "right": -0.510}  # clear of the stack's end faces by 105 and 80 mm

# where each hand takes the picked tote (its own end's bar) and where it ends
PICK_Y = {"left": L.PICK_Y + L.BAR_Y, "right": L.PICK_Y - L.BAR_Y}  # 0.415, 0.025
STACK_Y = {"left": L.BASE_Y + L.BAR_Y, "right": L.BASE_Y - L.BAR_Y}  # -0.025, -0.415


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, YAW)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def hand_plan(side: str, dx: float = 0.0, dy: float = 0.0, hold: bool = False):
    """One hand's moves: over its end's grip bar, down round it, grip, lift, carry across, lower onto the lower tote,
    open, up, aside. ``dx``, ``dy``: where the stack is put down, off the nest (counterfactual only); ``hold``:
    never let go (counterfactual only)."""
    yp, ys = PICK_Y[side], STACK_Y[side] + dy
    xs = X + dx
    plan = [
        _w(2.5, X, yp, HOVER, OPEN_G, "over the end hand hold"),
        _w(1.5, X, yp, GRIP_Z_PICK + 0.020, OPEN_G, "down to just above the bar"),
        _w(1.5, X, yp, GRIP_Z_PICK, OPEN_G, "slowly down: the pads straddle the bar"),
        _w(0.5, X, yp, GRIP_Z_PICK, OPEN_G, "settle"),
        _w(1.5, X, yp, GRIP_Z_PICK, GRIP_F, "grip the bar"),
        _w(2.0, X, yp, CARRY, GRIP_F, "lift the tote out of its bay"),
        _w(CARRY_S, xs, ys, CARRY, GRIP_F, "carry it across over the lower tote"),
        _w(0.5, xs, ys, CARRY, GRIP_F, "settle"),
        _w(1.5, xs, ys, ABOVE_SEAT, GRIP_F, "down to just above the lower tote's walls"),
        _w(2.5, xs, ys, GRIP_Z_SEAT - PRESS, GRIP_F, "slowly down: the foot goes into the lower tote"),
        _w(0.5, xs, ys, GRIP_Z_SEAT - PRESS, GRIP_F, "settle"),
    ]
    if hold:
        return plan
    plan += [
        _w(1.5, xs, ys, GRIP_Z_SEAT - PRESS, OPEN_G, "let go"),
        _w(1.5, xs + RETREAT_DX, ys, RETREAT, OPEN_G, "straight up, the pad tips clear of the rim"),
        _w(2.5, xs + RETREAT_DX, PARK_Y[side], RETREAT, OPEN_G, "aside, over the rim"),
        _w(1.5, xs + RETREAT_DX, PARK_Y[side], RETREAT, CLOSED, "closing, clear of the totes"),
    ]
    return plan


# Wrong ways (counterfactuals): where the stack is put down (x, y off the nest), or never released.
VARIANT = {"offset_x": {"dx": 0.012}, "offset_y": {"dy": 0.012}, "hold": {"hold": True}}
VARIANTS = tuple(VARIANT)


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant. Both hands work on one clock, the same moves
    mirrored: left on the +y end bar, right on the -y end bar."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    kw = VARIANT.get(variant, {})
    return {side: hand_plan(side, **kw) for side in ("left", "right")}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      offset_x   put down 12 mm off in x: past the lead-in, it rides on the wall tops, 9 mm high (0.5 offline)
      offset_y   put down 12 mm off in y: the same (0.5 offline)
      hold       lowered into place and never let go (0.0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
