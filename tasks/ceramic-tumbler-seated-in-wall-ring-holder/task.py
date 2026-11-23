"""On the station table (a bathroom vanity top), real size: a fixed wall ring holder and a loose ceramic tumbler.

The holder is a chrome-plated ring (68 mm inside, 8 mm wide band, 3 mm thick) on an 88 mm strap from a flange screwed
to an 18 mm plywood backboard (400 x 300 mm) that is screwed to the table; the ring's top face is 90 mm over the
table at x 240 mm, y -140 mm. The tumbler is a hollow glazed-ceramic cup, 100 mm tall, 60 mm across at its base
widening evenly to 76 mm at the rim, 5 mm wall, 0.28 kg, standing mouth up at x 220 mm, y -300 mm in front of the right
arm. Goal: lower it base first through the ring until the ring meets the taper (50 mm up from its base: the base
then hangs 40 mm over the table), upright and centred, and let go with both hands 30 mm clear.

The tumbler's wall is a 32-sided frustum of thin tilted slabs over a floor disc, so the ring meets a sloped wall (a
real wedge, centring itself), not a step. The ring is a 32-sided ring of boxes. Both are mechkit compounds
(mechkit.py, next to this file); the holder is a fixed compound, the tumbler a loose one. No software rules: no weld,
bond or drive. The arms are read through rigstate.py (the let-go check).

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
from scoring import HOLDER, TUMBLER  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Hang the tumbler in the wall ring, mouth up, until the ring holds it, and let go."


# -- the parts --------------------------------------------------------------------------------------------------------
def _ring_boxes():
    n = L.RING_SIDES
    width = 2.0 * L.RING_R_OUT * math.tan(math.pi / n) * 1.05  # a little over the facet: neighbours overlap
    rc = L.RING_R_IN + L.RING_BAND / 2
    out = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        out.append(mk.box((rc * math.cos(th), rc * math.sin(th), L.RING_H - L.RING_T / 2),
                          (L.RING_BAND, width, L.RING_T), L.CHROME, ypr=(th, 0.0, 0.0)))
    return out


def holder_shapes():
    """The ring, its 88 mm strap, the flange the strap is screwed through, and the backboard they stand on.
    Base frame: the ring's centre on the table top."""
    out = _ring_boxes()
    strap_x0 = L.RING_R_OUT - 0.002  # overlaps the ring's outer edge by 2 mm
    strap_x1 = L.FLANGE_X - L.RING_X + 0.001  # and the flange by 1 mm
    out.append(mk.box(((strap_x0 + strap_x1) / 2, 0.0, L.RING_H - L.RING_T / 2),
                      (strap_x1 - strap_x0, L.STRAP_W, L.RING_T), L.CHROME))
    out.append(mk.box((L.FLANGE_X - L.RING_X + L.FLANGE_T / 2, 0.0, L.RING_H - L.FLANGE_H / 2 + 0.010),
                      (L.FLANGE_T, L.FLANGE_W, L.FLANGE_H), L.CHROME))
    out.append(mk.box((L.BOARD_X0 - L.RING_X + L.BOARD_T / 2, L.BOARD_Y - L.RING_Y, L.BOARD_H / 2),
                      (L.BOARD_T, L.BOARD_W, L.BOARD_H), L.BOARD_WHITE))
    return out


def tumbler_shapes():
    """A hollow cup: a floor disc and 32 thin slabs leaning out 4.57 degrees (8 mm over 100 mm). Each slab's outer face
    runs from radius 30 mm at the table to 38 mm at the rim. Base frame: the centre of the underside, z up."""
    n = L.RING_SIDES
    a = L.TUM_ALPHA
    length = L.TUM_H / math.cos(a)
    width = 2.0 * (L.TUM_R_BASE + L.TUM_TAN * L.TUM_H) * math.tan(math.pi / n) * 1.06
    mid = L.TUM_H / 2
    rc = L.TUM_R_BASE + L.TUM_TAN * mid - L.TUM_WALL / 2 * math.cos(a)
    zc = mid + L.TUM_WALL / 2 * math.sin(a)
    out = [mk.cylinder((0.0, 0.0, L.TUM_FLOOR / 2), L.TUM_R_BASE - 0.0005, L.TUM_FLOOR, color=L.TUM_WHITE)]
    for i in range(n):
        th = 2.0 * math.pi * i / n
        out.append(mk.box((rc * math.cos(th), rc * math.sin(th), zc), (L.TUM_WALL, width, length), L.TUM_WHITE,
                          ypr=(th, a, 0.0)))
    return out


HOLDER_PART = mk.assembly(HOLDER, origin=(L.RING_X, L.RING_Y, L.TOP), shapes=holder_shapes(), color=L.CHROME,
                          friction=L.RING_FRICTION, tight_fit=True)
TUMBLER_PART = mk.assembly(TUMBLER, origin=(L.TUM_START[0], L.TUM_START[1], L.TOP), mass=L.TUM_MASS,
                           shapes=tumbler_shapes(), color=L.TUM_WHITE, friction=L.TUM_FRICTION, tight_fit=True)
OBJECTS = [HOLDER_PART, TUMBLER_PART]
SPECS = {TUMBLER: TUMBLER_PART, HOLDER: HOLDER_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part as an approximate box and drops it onto the highest box below (here the
# whole backboard's bounding box). Offline only (never in the simulator), this model gives the scorer: the tumbler
# stands where it started until a closed hand grips its upper wall (opening under GRIP_OPEN_MAX, the TCP within
# 12 mm of its axis, the pad tips 40 to 75 mm up its wall); it then rides with that TCP, stopped from below by the
# ring (the wall against the ring's inner edge: layout.seat_floor; on the ring's top when more than 4 mm off-centre,
# or on the table when it has been carried past). Let go (the hand opened) it falls straight down to the ring's wedge
# when within 4 mm of the ring's axis, else to the table. Always upright and still. No friction or tipping.
GRIP_OPEN_MAX = 0.80  # the fingers on the 76 mm rim read 76/93.9 = 0.81 closed on it; the demo commands 0.72
GRIP_XY = 0.012
GRIP_TIP_LO, GRIP_TIP_HI = 0.040, 0.075
ON_RING_E = L.ON_RING_NEAR  # the base disc still overlaps the ring's band up to this offset (71 mm)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = np.array([L.TUM_START[0], L.TUM_START[1], 0.0])  # base centre, table top at z 0
        self.grip: dict = {}  # arm -> (TCP to base offset) while it holds the tumbler

    def _offset(self) -> float:
        return float(np.hypot(self.pos[0] - L.RING_X, self.pos[1] - L.RING_Y))

    def floor_held(self) -> float:
        e = self._offset()
        f = L.seat_floor(e)
        if f is not None:
            return f
        return L.RING_H if e < ON_RING_E else 0.0

    def floor_free(self) -> float:
        f = L.seat_floor(self._offset())
        return 0.0 if f is None else f

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            rel = np.array([tcp[0], tcp[1], tcp[2] - L.TOP])
            tip_up = float(tips[2]) - L.TOP - float(self.pos[2])
            near = float(np.hypot(rel[0] - self.pos[0], rel[1] - self.pos[1])) <= GRIP_XY
            if opening < GRIP_OPEN_MAX and side not in self.grip and near and GRIP_TIP_LO <= tip_up <= GRIP_TIP_HI:
                self.grip[side] = self.pos - rel
            if opening >= GRIP_OPEN_MAX:
                self.grip.pop(side, None)
        if self.grip:
            side = next(iter(self.grip))
            tcp = read[rigstate.ARMS.index(side)][0]
            rel = np.array([tcp[0], tcp[1], tcp[2] - L.TOP])
            self.pos = rel + self.grip[side]
            self.pos[2] = max(self.pos[2], self.floor_held())
        else:
            self.pos[2] = self.floor_free()

    def states(self) -> dict:
        t = mk.state_from_poses(TUMBLER_PART, (self.pos[0], self.pos[1], L.TOP + self.pos[2], 0.0, 0.0, 0.0, 1.0),
                                source="offline")
        h = mk.state_from_poses(HOLDER_PART, (L.RING_X, L.RING_Y, L.TOP, 0.0, 0.0, 0.0, 1.0), source="offline")
        return {TUMBLER: t, HOLDER: h}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(TUMBLER), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, base mm, tilt deg, off-centre mm, hands gap mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: the tumbler's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The tumbler hanging in the ring (base 32 to 48 mm over the table), upright within 6 degrees, its axis within 6 mm
    of the ring's centre, both grippers 30 mm clear of it and the holder, still for 1 s, no latched hit; held 1 s.
    This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline model)."""
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
    row["source"] = getattr(parts.get(TUMBLER), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["base_mm"], d["tilt_deg"], d["off_centre_mm"], d["hands_gap_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the tumbler ever hit the ring or the table (base down 20 mm or more within 0.15 s, still 130 mm/s or faster
    in the last 0.05 s, then stopped dead on a support), or the holder moved over 5 mm, or unless both
    hands are 30 mm clear and the tumbler is still for 1 s; 1.0 hung (base 32 to 48 mm up), tilt <= 6 degrees,
    off-centre <= 6 mm; 0.5 hung, tilt <= 15 degrees, off-centre <= 12 mm; 0.25 caught on or in the ring (base 20 mm or
    more up, within 60 mm of its axis, mouth up); 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
GRIP = -(L.TUM_D_RIM - 0.008) / 2  # 68 mm commanded across the 76 mm rim
YAW = 0.0  # the fingers close along y
GRIP_TIPS_UP = 0.062  # pad tips 62 mm up the wall: the pads span 62 to 100 mm, up to the rim; 12 mm over the ring's
# top when hung (50 mm up the wall)
GRIP_TCP = GRIP_TIPS_UP + PAD_TIP  # TCP over the base: 75 mm
CARRY_BASE = L.RING_H + 0.015  # the base carried 15 mm over the ring's top
CARRY_TCP = CARRY_BASE + GRIP_TCP  # 180 mm
ENTRY_BASE = L.RING_H + 0.005  # the base 5 mm over the ring's top before it goes through the hole (GPU run 17 Nov:
# a 1.5 s descent straight through let the arm drift 5 mm off the hole's axis and the base caught the ring's edge)
FAST_BASE = L.H_SEAT + 0.012  # fast down to 12 mm over the seat; the last 12 mm are slow
PRESS = 0.003  # commanded this far past the seat: the ring stops the tumbler, the arm gives
HOVER = 0.150
CLEAR_TCP = 0.170  # up after the let-go: the pad tips (13 mm lower) end 17 mm over the hung tumbler's rim
RETREAT_X = 0.10  # then back 140 mm from the ring: the finger zones end more than 30 mm clear
HIGH_LET_GO = 0.030  # variant: let go this far over the seat


def _w(d, x, y, z, finger, label, yaw=YAW):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def plan(variant: str = "") -> list[Waypoint]:
    """The right arm's plan (the left arm stays at its ready pose). Pick the tumbler by its upper wall, carry it over
    the ring, lower it mouth up through the ring until the taper stops it, let go, rise clear, back away."""
    tx, ty = L.TUM_START
    rx, ry = L.RING_X, L.RING_Y
    seat_tcp = L.H_SEAT + GRIP_TCP
    low = seat_tcp - PRESS
    if variant == "high":
        low = seat_tcp + HIGH_LET_GO  # lets go 30 mm over the seat: it drops onto the ring
    wps = [
        _w(2.0, tx, ty, HOVER, OPEN, "over the tumbler"),
        _w(1.0, tx, ty, GRIP_TCP + 0.030, OPEN, "down to just above the grip"),
        _w(1.5, tx, ty, GRIP_TCP, OPEN, "slowly down, pads either side of the wall"),
        _w(0.5, tx, ty, GRIP_TCP, OPEN, "settle"),
        _w(1.5, tx, ty, GRIP_TCP, GRIP, "grip the upper wall"),
        _w(0.5, tx, ty, GRIP_TCP, GRIP, "settle the grip"),
        _w(1.5, tx, ty, CARRY_TCP, GRIP, "lift it straight up, base over the ring's height"),
        _w(2.0, rx, ry, CARRY_TCP, GRIP, "carry it over the ring"),
        _w(1.0, rx, ry, CARRY_TCP, GRIP, "settle over the ring (the base has 4 mm to spare in the hole)"),
        _w(1.0, rx, ry, ENTRY_BASE + GRIP_TCP, GRIP, "down to just over the ring's top"),
        _w(1.0, rx, ry, ENTRY_BASE + GRIP_TCP, GRIP, "settle, centred over the hole"),
        _w(2.5, rx, ry, FAST_BASE + GRIP_TCP, GRIP, "slowly down, base through the ring's hole"),
    ]
    if variant == "high":
        wps.append(_w(1.0, rx, ry, low, GRIP, "down to 30 mm over the seat"))
    else:
        wps.append(_w(2.0, rx, ry, low, GRIP, "slowly down until the taper meets the ring, press 3 mm"))
    wps.append(_w(0.5, rx, ry, low, GRIP, "settle"))
    if variant == "keep":
        wps.append(_w(4.0, rx, ry, low, GRIP, "keep holding it"))
        return wps
    wps += [
        _w(1.5, rx, ry, low, OPEN, "let go"),
        _w(1.5, rx, ry, CLEAR_TCP, OPEN, "straight up, pads clear of the rim"),
        _w(2.0, RETREAT_X, ry, CLEAR_TCP, OPEN, "back away from the ring"),
    ]
    return wps


# Wrong ways (counterfactuals), for the README and the tests: never the submitted demo.
VARIANTS = ("high", "keep")


def demo():
    """The right arm alone does the job; the left stays at its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold
    the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      high  lets go 30 mm over the seat: the tumbler drops onto the ring (a 30 mm drop in one step: 0)
      keep  lowers it into the ring and keeps holding it (not hands off: 0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    plans = {"right": plan(variant)}
    plans.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plans
