"""On the station table, real size: a stock MUJI stacking chest, half, 1 drawer, oak veneer (JAN 4550512381105,
370 x 280 x 185 mm) stands with its front at x 300 mm, centred on y 0, screwed down. Its drawer (inset front, an IKEA
BAGGANAS 143 mm bar handle screwed on) is shut and child-proofed with a Safety 1st cabinet and drawer latch (48390):
the latch screwed to the back of the drawer front 140 mm left of centre, its catch under the chest's top. Pulled, the
drawer stops with the back of its front 38 mm out of the carcass and the hook on the catch; pressing the latch arm
down through that gap draws the hook clear.

Goal: the drawer open 100 mm or more (the back of its front past the carcass's front edge), the latch and catch undamaged, both hands off.

The unit is one fixed mechkit articulation (mechkit.py, next to this file): the carcass, the drawer on a prismatic
joint, and the latch arm on a sprung revolute joint on the drawer. The catch is its own fixed part (links of one
assembly never collide with each other, and the hook must meet the catch). There is no software rule: the drawer is
held by the hook's face against the catch's lip and released by pressing the arm. The arms are read through
rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import CATCH, DRAWER, LATCH, UNIT  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Open the child-proofed drawer at least 100 mm and let go."

# -- the parts --------------------------------------------------------------------------------------------------------
DRAWER_LINK = mk.slider(DRAWER, "-X", P.drawer_shapes(), upper=L.TRAVEL, lower=0.0, anchor=(0.0, 0.0, 0.0),
                        mass=L.DRAWER_MASS, damping=L.DRAWER_DAMPING, friction=0.0, color=L.PLY)
LATCH_LINK = mk.hinge(LATCH, "Y", L.HINGE, P.latch_shapes(), upper=L.LATCH_UPPER, lower=L.LATCH_LOWER,
                      mass=L.LATCH_MASS, spring=L.LATCH_SPRING, damping=L.LATCH_DAMPING, friction=0.0,
                      parent=DRAWER, color=L.LATCH_WHITE)
UNIT_PART = mk.assembly(UNIT, origin=L.ORIGIN, shapes=P.carcass_shapes(), joints=[DRAWER_LINK, LATCH_LINK],
                        color=L.PLY, friction=(0.5, 0.4), solver_iterations=32)
CATCH_PART = mk.assembly(CATCH, origin=L.ORIGIN, shapes=P.catch_shapes(), color=L.LATCH_WHITE, friction=(0.4, 0.3),
                         tight_fit=True)
OBJECTS = [UNIT_PART, CATCH_PART]


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy only carries loose parts; the unit is fixed and has no hook. Offline only (never in the
# simulator), this model gives the scorer: (1) the drawer following a gripper closed on the handle bar (the pads round
# the bar), stopped at a 38 mm gap (53 mm of travel) while the hook is up and behind the lip, riding the lip's
# underside between 38 and 50 mm of gap;
# (2) the latch arm pressed down by closed pads that are in the gap between the drawer front and the carcass, over the
# arm; it springs back when they leave. Nothing else moves the drawer. The press is not taken as exact geometry: the
# tilting arm meets the pads' edge nearest its pivot, and the arm gives (its joints are PD springs), so the pads sink
# only until the spring's push balances that give (pressed_angle). The first GPU run (17 Nov) pressed 5 mm from the
# pivot with the tips commanded 3 mm into the arm and reached 2.5 deg, where geometry alone predicted 0.21 rad.
GRIP_TOL_X = 0.006  # the TCP within 6 mm of the bar's centre along x
BAR_GAP = L.HANDLE_W + 0.004  # pads closed to within 4 mm of the bar's 9 mm: gripped
CLOSED_GAP = 0.012  # pads closed: they press as one finger
TIP_HALF = 0.006  # the closed pair is about 12 mm thick at the tip (two 6 mm fingers; the GPU contact fits it)
ARM_GIVE = 0.0025  # m/N: the pad tips' vertical give at the press pose (J K^-1 J^T with the PD gains: 1.9 mm/N; the
# 17 Nov GPU press fits 1.6 to 2.4 mm/N). The latch's own weight, which helps the press, is left out.

# The pads against the rigid parts (offline only). GPU run 3 (17 Nov, the MUJI chest) scored 0 because the right pads
# came down on the chest's top on the way to the handle and stayed there; the stand-in above let a gripper pass
# through anything, so it scored that demo 1.0. Now each pad is a box (both arms close along world x: YAW_R, YAW_L
# are +-90 deg; the grip is taken as straight down), 6 mm thick, 21 mm wide over its lowest 6.5 mm and 27 mm above
# (15 mm at the tip, 27 mm at the TCP: research/sim_capabilities.md), from the tips to 25 mm above the TCP. A pad
# more than HIT_TOL into the chest, the drawer (front, sides, handle, the latch's foot) or the catch along every axis
# rests on that part: from then on the arm is blocked (it neither grips nor presses), as on the GPU, where the arm,
# a soft spring pushed onto a rigid part, stalls there and friction holds it.
PAD_T = TIP_HALF
PAD_TIP_H, PAD_HALF_W_TIP, PAD_HALF_W, PAD_UP = 0.0065, 0.0105, 0.0135, 0.025
HIT_TOL = 0.001


def _world_boxes(shapes, tag) -> list:
    out = []
    for i, sh in enumerate(shapes):
        c = np.asarray(sh.center, dtype=float) + np.asarray(L.ORIGIN, dtype=float)
        h = np.asarray(sh.aabb_half(np.eye(3)), dtype=float)
        out.append((f"{tag}[{i}]", c - h, c + h))
    return out


FIXED_BOXES = _world_boxes(P.carcass_shapes(), "chest") + _world_boxes(P.catch_shapes(), "catch")
_N_DRAWER_BODY = len(P.drawer_shapes()) - len(P.handle_shapes())
DRAWER_BOXES = (_world_boxes(P.drawer_shapes()[:_N_DRAWER_BODY], "drawer")
                + _world_boxes(P.handle_shapes()[:1], "handle bar") + _world_boxes(P.handle_shapes()[1:], "handle leg"))


def pad_boxes(tcp, tips, gap) -> list:
    """World boxes (lo, hi) of both pads for a TCP, its pad-tip point and the pads' gap (closing along world x)."""
    tcp, tips = np.asarray(tcp, dtype=float), np.asarray(tips, dtype=float)
    out = []
    for sign in (-1.0, 1.0):
        xc = tcp[0] + sign * (gap / 2 + PAD_T / 2)
        for z0, z1, hw in ((tips[2], tips[2] + PAD_TIP_H, PAD_HALF_W_TIP),
                           (tips[2] + PAD_TIP_H, tcp[2] + PAD_UP, PAD_HALF_W)):
            out.append((np.array([xc - PAD_T / 2, tcp[1] - hw, z0]), np.array([xc + PAD_T / 2, tcp[1] + hw, z1])))
    return out


def pad_hit(tcp, tips, gap, o: float, skip=()) -> tuple | None:
    """(part, depth m) of the deepest rigid part a pad is in by more than HIT_TOL along every axis, drawer at travel
    ``o``; None if the pads are clear. ``skip``: part-name prefixes left out (the bar a gripper holds)."""
    shift = np.array([-o, 0.0, 0.0])
    worst = None
    for lo, hi in pad_boxes(tcp, tips, gap):
        for name, blo, bhi in FIXED_BOXES + [(n, a + shift, b + shift) for n, a, b in DRAWER_BOXES]:
            if any(name.startswith(k) for k in skip):
                continue
            depth = float(np.min(np.minimum(hi, bhi) - np.maximum(lo, blo)))
            if depth > HIT_TOL and (worst is None or depth > worst[1]):
                worst = (name, depth)
    return worst


def pressed_angle(r: float, z: float, give: float = ARM_GIVE) -> float:
    """The latch angle at which pads commanded to height ``z`` (base frame), touching the arm ``r`` from its pivot,
    stop: the arm's top at the pads' height plus their give under the spring's push (0 if they do not reach it)."""
    if r <= 0.0 or z >= L.arm_top_z(r, 0.0):
        return 0.0

    def short(a):  # how far the arm's top is still above the pads where they come to rest
        return L.arm_top_z(r, a) - (z + give * L.LATCH_SPRING * a / r)

    if short(L.LATCH_UPPER) >= 0.0:
        return L.LATCH_UPPER
    lo, hi = 0.0, L.LATCH_UPPER
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if short(mid) > 0.0 else (lo, mid)
    return (lo + hi) / 2


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.o = 0.0
        self.a = 0.0
        self.hold: dict = {}  # arm side -> TCP x minus the bar's x when gripped
        self.blocked: dict = {}  # arm side -> (part, depth mm, step): its pads came down on a rigid part
        self.steps = 0

    def bar_x(self) -> float:
        return L.UX + L.HANDLE_XC - self.o  # world x of the bar's centre

    def _on_bar(self, tcp, tips, gap) -> bool:
        y_ok = abs(float(tcp[1]) - L.UY) <= L.HANDLE_HOLES / 2 - 0.010
        z_ok = float(tips[2]) - L.TOP <= L.HANDLE_Z - L.HANDLE_W / 2 + 0.002 and \
            float(tips[2]) - L.TOP + rigstate.FINGER_LEN >= L.HANDLE_Z + L.HANDLE_W / 2
        return bool(gap <= BAR_GAP and abs(float(tcp[0]) - self.bar_x()) <= GRIP_TOL_X and y_ok and z_ok)

    def _press(self, tcp, tips, gap) -> float:
        """The latch angle these pads impose (0 if they do not reach the arm)."""
        if gap > CLOSED_GAP:
            return 0.0
        x = float(tips[0]) - L.UX  # base frame
        if not (L.FRONT_BACK - self.o + TIP_HALF <= x <= -TIP_HALF):  # not in the gap: they stand on the front/top
            return 0.0
        if abs(float(tips[1]) - L.UY - L.LATCH_Y) > L.LATCH_HALF_W + rigstate.FINGER_HALF_WIDTH / 2:
            return 0.0
        r = x + self.o - TIP_HALF - L.HINGE[0]  # the pads' edge nearest the pivot line, along the drawer
        if r <= 0.002:
            return 0.0
        return pressed_angle(r, float(tips[2]) - L.TOP)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        self.steps += 1
        current = {side: (tcp, tips, opening * rigstate.FULL_GAP)
                   for side, (tcp, tips, opening) in zip(rigstate.ARMS, read)}
        for side, (tcp, tips, gap) in current.items():
            if side in self.blocked:
                continue
            hit = pad_hit(tcp, tips, gap, self.o, skip=("handle bar",) if side in self.hold else ())
            if hit is not None:
                self.blocked[side] = (hit[0], round(hit[1] * 1000, 1), self.steps)
                self.hold.pop(side, None)
        free = {side: v for side, v in current.items() if side not in self.blocked}
        a = max((self._press(*v) for v in free.values()), default=0.0)
        for side, (tcp, tips, gap) in free.items():
            if side in self.hold:
                if gap > BAR_GAP + 0.004:
                    del self.hold[side]
                continue
            if self._on_bar(tcp, tips, gap):
                self.hold[side] = float(tcp[0]) - self.bar_x()
        o = self.o
        if self.hold:
            side, off = next(iter(self.hold.items()))
            want = L.UX + L.HANDLE_XC - (float(current[side][0][0]) - off)
            o = float(np.clip(want, 0.0, L.TRAVEL))
        if a < L.CLEAR_RAD:
            if self.o <= L.ENGAGE_AT + 1e-6 and o > L.ENGAGE_AT:
                o = L.ENGAGE_AT  # the hook's face on the lip
            if L.ENGAGE_AT < o < L.PAST_AT:
                a = L.CLEAR_RAD  # the hook rides on the lip's underside
        self.o, self.a = o, a

    def states(self) -> dict:
        q = (0.0, 0.0, 0.0, 1.0)
        return {UNIT: mk.state_from_poses(UNIT_PART, (*L.ORIGIN, *q), joints={DRAWER: self.o, LATCH: self.a},
                                          source="offline"),
                CATCH: mk.state_from_poses(CATCH_PART, (*L.ORIGIN, *q), source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(UNIT), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, opening mm, latch deg]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: the joints and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The drawer open 100 mm or more, never forced through the catch, both grippers 30 mm clear and the drawer still
    for 1 s; held 1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the
    offline stand-in)."""
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
    row["source"] = getattr(parts.get(UNIT), "source", None)
    if row["source"] == "offline" and _OFFLINE.blocked:
        row["offline_blocked"] = dict(_OFFLINE.blocked)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["opening_mm"], d["latch_deg"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the hook was ever forced through the catch, or the drawer is open less than 60 mm; 1.0 open 100 mm or
    more with both hands off and the drawer still for 1 s; 0.5 open 60 to 100 mm, hands off and still; 0.25 open
    60 mm or more otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap. Drawer positions (TRY, EASE, CLEAR,
# FULL) are the slider's travel: the gap behind the inset front is 15 mm less (FB).
OPEN = -0.04695  # 94 mm
STRADDLE = -0.011  # 22 mm: the pads straddle the 9 mm bar 6.5 mm clear each side, the rear pad in the 22 mm
# behind it 6.5 mm clear of the front
GRIP_BAR = 0.0  # closed on the bar
CLOSED = 0.0
YAW_R = np.pi / 2  # the right pads close along x, front and back of the bar
YAW_L = -np.pi / 2  # the left pair's 8 mm tip thickness along x, in the 35 to 38 mm gap
TIP = 0.013  # TCP above the pad tips

FB = L.FRONT_BACK  # 15 mm: the inset front's back face inside the carcass when shut
BAR_X0 = L.UX + L.HANDLE_XC  # 273.5 mm: the bar's centre, drawer shut
GRIP_Y = -0.042  # the right hand on the bar 42 mm right of centre: its pads, 27 mm wide at the TCP, stay 4.5 mm
# clear of the leg at -64 mm (r 4 mm). GPU run 3 used -50 mm, where the rear pad's corner comes down 3.5 mm onto the
# leg; -40 mm (run 2) leaves 6.5 mm but brings the forearms to 25 mm apart (capsules) with the latch at y 140 mm
Z_BAR = L.HANDLE_Z - L.HANDLE_W / 2 - 0.0055 + TIP  # pad tips 5.5 mm below the bar's underside
HOVER_R = 0.200
APPROACH_R = Z_BAR + 0.037  # 180 mm: over the handle, the pad tips 22 mm above the bar and 18 mm BELOW the chest's
# top (185 mm). Over the bar the hand cannot go higher than 195 mm (the straight-down reach limit, joint 4), so its
# pads can never pass over the chest's top there: they must come in from in front of the chest (FRONT_R).
FRONT_R = (BAR_X0 - 0.035, GRIP_Y, 0.190)  # first stop, 35 mm in front of the bar, the pads closed to straddle on
# the way: the rear pad 45 mm in front of the carcass's edge, the tips 22 mm above the bar. On the way there from the
# ready pose (TCP x 323 mm, over the chest) the rear pad crosses the carcass's edge with its tip 18 mm above the top
# (9 mm if the arm's x lags its z as on run 3). GPU run 3 went from the ready pose straight to over the handle while
# the fingers were still closing: the rear pad came down on the chest's top (TCP stalled at x 287 to 296, z 197 mm
# from step 43 to 127, commanded down to 143), so the right hand never got to the bar.
TRY = FB + 0.042  # the first pull: the gap 4 mm past the stop at 38 mm (the hook takes it)
EASE = FB + 0.035  # eased back 3 mm so the hook's face is off the lip before the press (gap 35 mm)
PRESS_IN = 0.024  # the pads' centre 24 mm behind the back of the front at a 35 mm gap: their front faces 5 mm clear of
# the carcass's edge, their edge nearest the pivot 13 mm from it (the gap's middle, used first, put it at 5 mm)
PRESS_X = L.UX + FB - EASE + PRESS_IN
PRESS_DEPTH = 0.012  # the pad tips commanded 12 mm below the arm's top: the arm's give leaves about 0.17 rad (twice
# the 0.084 rad that draws the hook clear) at 2.5 mm/N, 0.11 rad at 4 mm/N; a stiffer arm only meets the stop (0.25 rad)
Z_PRESS = L.ARM_Z1 - PRESS_DEPTH + TIP
Z_ABOVE = L.ARM_Z1 + 0.009 + TIP  # 9 mm above the arm
HOVER_L = 0.195  # 20 mm under the reach limit over the latch; the closed pads cross to it 5 mm in front of the
# carcass's edge with their tips 3 mm under its top (run 3: left TCP x within 0.5 mm of its target there)
CLEAR = FB + 0.065  # the left hand lets go of the arm with the hook 15 mm past the lip (gap 65 mm)
FULL = 0.150  # pulled 150 mm (the gap 135 mm)
PULL_SPEED = 0.030  # m/s
THROUGH_SPEED = 0.015  # m/s while the hook passes under the catch: half the drag, so the pulling hand lags the
# pressing one less and the pads stay where they press
WAIT_L = (0.27, 0.20, HOVER_L)  # 70 mm or more from the right arm while it goes down to the handle (capsules)
PARK_L = (0.24, 0.30, 0.180)  # beside the chest, 115 mm off its side
PARK_R = (0.20, -0.28, 0.180)  # beside the open drawer, 100 mm from it


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def right_plan(variant: str = "") -> list:
    """The right arm: come in from in front of the chest (never over its top), grip the handle, pull to the catch,
    ease off 3 mm, wait for the press, pull with the left hand to a 65 mm gap, wait for it to leave, pull on to 150 mm
    of travel, let go and stand clear."""
    x0, y = BAR_X0, GRIP_Y
    plan = [
        _w(2.0, *FRONT_R, STRADDLE, "in front of the chest, pads to straddle", YAW_R),
        _w(1.2, x0, y, APPROACH_R, STRADDLE, "over the handle", YAW_R),
        _w(1.2, x0, y, Z_BAR + 0.020, STRADDLE, "down in front of the drawer, the rear pad behind the bar", YAW_R),
        _w(1.0, x0, y, Z_BAR, STRADDLE, "slowly down round the bar", YAW_R),
        _w(0.3, x0, y, Z_BAR, STRADDLE, "settle", YAW_R),
        _w(1.0, x0, y, Z_BAR, GRIP_BAR, "grip the bar", YAW_R),
        _w(1.5, x0 - TRY, y, Z_BAR, GRIP_BAR, "pull: the drawer stops on the catch", YAW_R),
        _w(0.5, x0 - TRY, y, Z_BAR, GRIP_BAR, "hold against the catch", YAW_R),
        _w(1.0, x0 - EASE, y, Z_BAR, GRIP_BAR, "ease off 3 mm", YAW_R),
    ]
    if variant == "no_press":
        plan += [_w((FULL - EASE) / PULL_SPEED, x0 - FULL, y, Z_BAR, GRIP_BAR, "pull hard without the press", YAW_R),
                 _w(0.5, x0 - FULL, y, Z_BAR, GRIP_BAR, "settle", YAW_R)]
    else:
        plan += [
            _w(5.0, x0 - EASE, y, Z_BAR, GRIP_BAR, "hold while the left hand presses the latch", YAW_R),
            _w((CLEAR - EASE) / THROUGH_SPEED, x0 - CLEAR, y, Z_BAR, GRIP_BAR, "pull past the catch", YAW_R),
            _w(1.5, x0 - CLEAR, y, Z_BAR, GRIP_BAR, "hold while the left hand leaves", YAW_R),
        ]
        end = FULL if variant != "half_open" else FB + 0.080
        plan += [
            _w((end - CLEAR) / PULL_SPEED, x0 - end, y, Z_BAR, GRIP_BAR, "pull the drawer open", YAW_R),
            _w(0.5, x0 - end, y, Z_BAR, GRIP_BAR, "settle", YAW_R),
        ]
    end = plan[-1].pos[0]
    if variant == "hold_on":
        return plan
    plan += [
        _w(1.0, end, y, Z_BAR, STRADDLE, "let go of the bar", YAW_R),
        _w(1.0, end, y, HOVER_R, STRADDLE, "straight up in front of the drawer", YAW_R),
        _w(1.5, *PARK_R, STRADDLE, "aside", YAW_R),
    ]
    return plan


def left_plan(variant: str, right: list) -> list:
    """The left arm: wait over the latch, go down into the gap once the drawer is eased off the catch, press the arm
    down, move out with the drawer to a 65 mm gap, let the latch spring back and stand clear."""
    x, y = PRESS_X, L.UY + L.LATCH_Y
    plan = [_w(2.5, *WAIT_L, CLOSED, "pads closed, wait clear of the right arm", YAW_L)]
    eased = _until(right, "ease off")
    plan.append(_w(eased - _duration(plan), *WAIT_L, CLOSED, "wait for the gap", YAW_L))
    plan += [
        _w(1.0, x, y, HOVER_L, CLOSED, "over the latch", YAW_L),
        _w(1.0, x, y, Z_ABOVE + 0.020, CLOSED, "down into the gap", YAW_L),
        _w(1.0, x, y, Z_ABOVE, CLOSED, "slowly down to just above the latch arm", YAW_L),
        _w(1.0, x, y, Z_PRESS, CLOSED, "press the latch arm down", YAW_L),
        _w(1.0, x, y, Z_PRESS, CLOSED, "hold it down", YAW_L),
    ]
    dt = (CLEAR - EASE) / THROUGH_SPEED
    plan.append(_w(dt, x - (CLEAR - EASE), y, Z_PRESS, CLOSED, "keep it down while the drawer comes out", YAW_L))
    plan += [
        _w(0.5, x - (CLEAR - EASE), y, Z_PRESS, CLOSED, "settle", YAW_L),
        _w(1.0, x - (CLEAR - EASE), y, HOVER_L, CLOSED, "straight up: the latch springs back", YAW_L),
        _w(1.5, *PARK_L, OPEN, "aside", YAW_L),
    ]
    return plan


# Wrong ways (counterfactuals), each with the score it earns.
VARIANTS = ("no_press", "half_open", "hold_on")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the right hand grips the handle and pulls; the drawer stops on the catch with a 38 mm gap behind
    its front; the right hand eases off 3 mm; the left hand goes down into the gap and presses the latch arm; both
    move out together to a 65 mm gap (the hook passes under the catch); the left hand leaves; the right pulls on to
    150 mm of travel and lets go."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = right_plan(variant)
    if variant == "no_press":
        return {"right": right}
    return {"left": left_plan(variant, right), "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_press   the right hand pulls to 150 mm with no press: the hook holds the drawer at a 38 mm gap: 0.0
      half_open  pressed and pulled to an 80 mm gap only: 0.5
      hold_on    pulled to 150 mm but the right hand keeps hold of the handle: 0.25"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
