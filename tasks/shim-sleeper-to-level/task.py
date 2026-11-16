"""On the station table, real size: two fixed 4x4 pads (89 x 89 mm, 300 mm long along x) 370 mm apart on y, the near
one (y -10 mm) standing 6.0 mm lower than the far one (y -380 mm), as on a floor that dips. A loose 2x4 floor sleeper
(38 x 89 x 610 mm, 0.87 kg) stands on edge across them along y, so it slopes 9.9 mm end to end, low at the +y end. A
9 in torpedo level lies along its top. On the low pad, along x, lies a wood shim (8 in x 1-1/2 in, 3/8 in tapering
to 1/16 in), its thin tip 2 mm short of the sleeper's front face, its butt toward the robot.

Goal: the sleeper level (end to end within 1 mm over its 610 mm), the shim crossing its whole width under the low
end, the level left on it, hands off. The shim can only go in with the low end lifted off it: the tip is 1.6 mm
thick and the sleeper's underside there is 0.4 to 1.0 mm off the pad. The sleeper lies flush on the sloped shim, so
its height is the shim's thickness at the sleeper's rear face: pushed in 151 mm past the front face (6.0 mm thick at the
rear face) it carries the low end level. Pushed further it lifts the end past level (fully in: 3.1 mm high).

The sleeper, the shim, the level and the pads are mechkit compounds (mechkit.py, next to this file). One declared
software rule (``_bubble_rule``, a mechkit on_step callback): the level's bubble is a small link on a slider along
the vial, driven every control step to VIAL_R x the level's own measured pitch (toward the high end). It shows the
reading; the scorer reads the sleeper's pose, not the bubble. The arms are read through rigstate.py (the let-go
check).

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
from scoring import LEVEL, SHIM, SLEEPER  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Make the sleeper level with the shim, and leave the level on it."


# -- the parts --------------------------------------------------------------------------------------------------------
_RX0 = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(L.ROLL0), -math.sin(L.ROLL0)],
                 [0.0, math.sin(L.ROLL0), math.cos(L.ROLL0)]])
_UP0 = _RX0 @ np.array([0.0, 0.0, 1.0])


def _rolled(centre, size, color):
    """A box drawn turned ROLL0 about x (the slope the sleeper rests at as found), its centre given unturned."""
    c = _RX0 @ np.asarray(centre, dtype=float)
    return mk.box(tuple(float(v) for v in c), size, color, ypr=(0.0, 0.0, L.ROLL0))


def pad_shapes(height):
    return [mk.box((0.0, 0.0, height / 2), (L.PAD_LX, L.PAD_W, height), L.PAD_BROWN)]


def sleeper_shapes():
    return [_rolled((0.0, 0.0, 0.0), (L.SL_W, L.SL_L, L.SL_H), L.SPF)]


SLAB_PITCH = math.atan(L.SH_SLOPE)


def shim_top(x):
    """The shim's top above its underside (m) at ``x`` from its centre (tip toward +x)."""
    return (L.SH_BUTT + L.SH_TIP) / 2 - L.SH_SLOPE * x


def shim_shapes():
    """The taper as one slab turned so its top runs from 9.5 mm at the butt to 1.6 mm at the tip, with flat blocks
    under it standing on the underside (each as tall as the slab's underside at its thin end): no step pokes up
    through the sloped top. Base frame at the centre of the shim's underside, local x toward the tip."""
    h = L.SH_SLAB
    n = np.array([math.sin(SLAB_PITCH), 0.0, math.cos(SLAB_PITCH)])
    c = np.array([0.0, 0.0, shim_top(0.0)]) - n * h / 2
    out = [mk.box(tuple(float(v) for v in c), (L.SH_L / math.cos(SLAB_PITCH), L.SH_W, h), L.SHIM_WOOD,
                  ypr=(0.0, SLAB_PITCH, 0.0))]
    seg = L.SH_L / L.SH_FILLERS
    for i in range(L.SH_FILLERS):
        xa = -L.SH_L / 2 + i * seg
        xb = xa + seg
        z = shim_top(xb) - h / math.cos(SLAB_PITCH) - 0.0001
        if z >= 0.0004:
            out.append(mk.box(((xa + xb) / 2, 0.0, z / 2), (seg, L.SH_W, z), L.SHIM_WOOD))
    return out


def level_shapes():
    top = L.LV_H / 2
    out = [_rolled((0.0, 0.0, 0.0), (L.LV_W, L.LV_L, L.LV_H), L.LV_YELLOW),
           _rolled((0.0, 0.0, top + L.VIAL_T / 2), (L.VIAL_W, L.VIAL_L, L.VIAL_T), L.VIAL_GREEN)]
    for s in (-1, 1):
        out.append(_rolled((0.0, s * L.MARK_C, top + L.VIAL_T + 0.0004), (L.VIAL_W, L.MARK_W, 0.0008),
                           L.MARK_BLACK))
    return out


BUBBLE_Z = L.LV_H / 2 + L.VIAL_T + 0.0003 + L.BUBBLE_T / 2
BUBBLE = mk.slider("bubble", "Y", [_rolled((0.0, 0.0, BUBBLE_Z), (L.BUBBLE_W, L.BUBBLE_L, L.BUBBLE_T),
                                           L.BUBBLE_WHITE)],
                   upper=L.BUBBLE_MAX, lower=-L.BUBBLE_MAX, mass=L.BUBBLE_MASS, spring=20.0, damping=0.4)

SL_ORIGIN = (L.SL_X, L.SL_C0[0], L.TOP + L.SL_C0[1])
_LV_C = np.array(SL_ORIGIN) + _UP0 * (L.SL_H / 2 + L.LV_H / 2)
LV_ORIGIN = tuple(float(v) for v in _LV_C)

LOW_PAD = mk.assembly("low_pad", origin=(L.PAD_X, L.LOW_Y, L.TOP), shapes=pad_shapes(L.LOW_H), color=L.PAD_BROWN,
                      friction=L.FRICTION, tight_fit=True)
FAR_PAD = mk.assembly("far_pad", origin=(L.PAD_X, L.FAR_Y, L.TOP), shapes=pad_shapes(L.FAR_H), color=L.PAD_BROWN,
                      friction=L.FRICTION, tight_fit=True)
SLEEPER_PART = mk.assembly(SLEEPER, origin=SL_ORIGIN, mass=L.SL_MASS, shapes=sleeper_shapes(), color=L.SPF,
                           friction=L.FRICTION, tight_fit=True)
SHIM_PART = mk.assembly(SHIM, origin=(L.SH_X0, L.SH_Y, L.TOP + L.LOW_H), mass=L.SH_MASS, shapes=shim_shapes(),
                        color=L.SHIM_WOOD, friction=L.FRICTION, tight_fit=True)
LEVEL_PART = mk.assembly(LEVEL, origin=LV_ORIGIN, mass=L.LV_MASS, shapes=level_shapes(), joints=[BUBBLE],
                         color=L.LV_YELLOW, friction=L.LEVEL_FRICTION, tight_fit=True)
OBJECTS = [LOW_PAD, FAR_PAD, SLEEPER_PART, SHIM_PART, LEVEL_PART]
SPECS = {SLEEPER: SLEEPER_PART, SHIM: SHIM_PART, LEVEL: LEVEL_PART}


# -- the one declared rule: the bubble shows the level's own pitch ----------------------------------------------------
def bubble_target(level_state) -> float:
    """VIAL_R x the level's measured pitch along its long axis, toward the high end, stopped by the vial's ends; 0 when
    the level's state cannot be read."""
    try:
        r = level_state.rotation()
        a = r @ np.array([0.0, math.cos(L.ROLL0), math.sin(L.ROLL0)])
        if not np.all(np.isfinite(a)):
            return 0.0
        pitch = math.asin(max(-1.0, min(1.0, float(a[2])))) * (1.0 if a[1] >= 0 else -1.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return S.bubble_offset(pitch)


@mk.on_step
def _bubble_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    mk.set_targets(LEVEL_PART, "bubble", bubble_target(states.get(LEVEL)))


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it neither pivots the sleeper about
# the far pad nor pushes the shim. Offline only (never in the simulator), this model gives the scorer: (1) the shim
# pushed along +x by a closed gripper whose pad tips are on its butt face's height, held back where its thickness
# under the sleeper's front face would exceed the gap there (a jam); (2) the sleeper as a rigid line resting on the
# far pad and on the highest support at the low end (the pad's outer edge, or the shim by its thickness at the
# sleeper's rear face), or, held by a gripper at its low end, pivoting about the far pad with the hand; (3) the level
# riding on the sleeper.
PAD_FACE = 0.005  # a closed pad block's face ahead of the TCP along the closing direction, less the arm's trail
# (GPU-measured: the face 5.9 mm ahead of the TCP, the TCP 0.9 mm behind its command at the end of the push)
JAM_GIVE = 0.0  # a shim under a resting sleeper does not advance in this model (see OfflineModel)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.u = 0.0  # how far the shim has been pushed (m)
        self.grip: dict = {}  # arm -> offset of the sleeper's underside height at the grip from the TCP height
        self.line = (L.START_SLOPE, L.START_PIVOT)

    def d(self) -> float:
        return L.SH_TIP_X0 + self.u - L.SL_FRONT_X

    def _support_line(self):
        best = L.rest_on(0.0, L.LOW_Y1)
        a = L.shim_bear(self.d())
        if a > 0.0:
            y_c = L.SH_Y + L.SH_W / 2 if a < L.DIP else L.SH_Y - L.SH_W / 2
            cand = L.rest_on(a, y_c)
            if L.h_line(L.SL_Y1, *cand) > L.h_line(L.SL_Y1, *best):
                best = cand
        return best

    def _gap_at_shim(self, line) -> float:
        return min(L.h_line(L.SH_Y - L.SH_W / 2, *line), L.h_line(L.SH_Y + L.SH_W / 2, *line))

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        rest = self._support_line()
        line = rest
        if read is not None:
            for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
                y = float(tcp[1])
                h_now = L.h_line(y, *self.line)
                tip_z = float(tips[2]) - L.TOP
                holding = (opening < 0.6 and abs(float(tcp[0]) - L.SL_X) < 0.012 and L.SL_Y0 < y < L.SL_Y1
                           and L.LOW_H + h_now < tip_z < L.LOW_H + h_now + L.SL_H + 0.010)
                if not holding:
                    self.grip.pop(side, None)
                    continue
                if side not in self.grip:
                    self.grip[side] = h_now - (float(tcp[2]) - L.TOP)
                h = float(tcp[2]) - L.TOP + self.grip[side]
                if h > L.h_line(y, *rest):
                    slope, pivot = L.rest_on(h, y)
                    line = (slope, pivot)
        self.line = line
        if read is None:
            return
        for tcp, tips, opening in read:
            tip_z = float(tips[2]) - L.TOP - L.LOW_H
            if opening > 0.3 or not (-0.02 <= tip_z <= L.SH_BUTT - 0.001):
                continue
            if abs(float(tcp[1]) - L.SH_Y) > L.SH_W / 2 + rigstate.FINGER_HALF_WIDTH:
                continue
            butt = L.SH_BUTT_X0 + self.u
            face = float(tcp[0]) + PAD_FACE
            if face > butt and float(tcp[0]) < butt + 0.010:
                gap = self._gap_at_shim(self.line) + JAM_GIVE
                if gap < L.SH_TIP:
                    d_max = self.d()  # the tip cannot get under: it stops at the face
                    d_max = min(d_max, 0.0)
                else:
                    d_max = (gap - L.SH_TIP) / L.SH_SLOPE
                    d_max = max(d_max, self.d())  # never pulled back
                u_max = d_max - (L.SH_TIP_X0 - L.SL_FRONT_X)
                self.u = max(self.u, min(face - L.SH_BUTT_X0, u_max))
        self.line = self._support_line() if not self.grip else self.line

    def states(self) -> dict:
        slope, pivot = self.line
        phi = math.atan(slope)
        cy, cz = L.box_centre(slope, pivot)
        q = mk.quat_from_ypr(0.0, 0.0, phi - L.ROLL0)
        sl = mk.state_from_poses(SLEEPER_PART, (L.SL_X, cy, L.TOP + cz, *q), source="offline")
        sh = mk.state_from_poses(SHIM_PART, (L.SH_X0 + self.u, L.SH_Y, L.TOP + L.LOW_H, 0.0, 0.0, 0.0, 1.0),
                                 source="offline")
        r_rel = mk.quat_to_matrix(q)
        s0 = np.array(SL_ORIGIN)
        lv_origin = np.array([L.SL_X, cy, L.TOP + cz]) + r_rel @ (np.array(LV_ORIGIN) - s0)
        lv = mk.state_from_poses(LEVEL_PART, (*lv_origin, *q), source="offline")
        return {SLEEPER: sl, SHIM: sh, LEVEL: lv}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(SLEEPER), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, shim tip past front mm, shim yaw deg, diff mm,
# low end mm, bubble mm]


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
    """On the pads, the shim crossing the sleeper's whole width under its low end, the level on it, the sleeper end to
    end within 1 mm, both grippers 30 mm clear and every part still for 1 s, no latched fall; held 1 s. This runs
    first every control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
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
    try:
        row["bubble_joint_mm"] = round(float(parts[LEVEL].joints.get("bubble", 0.0)) * 1000.0, 2)
    except (AttributeError, KeyError, TypeError, ValueError):
        row["bubble_joint_mm"] = None
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(SLEEPER), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["shim_tip_past_front_mm"], d["shim_yaw_deg"], d["diff_mm"],
                      d["sleeper_low_end_mm"], d["bubble_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the sleeper or the level fell off the pads or the sleeper tipped onto its face, or unless every part is
    let go and still; 1.0 the shim crossing the sleeper's whole width with the sleeper within 1 mm end to end and the
    level on it; 0.5 the same within 2 mm; 0.25 the shim under the sleeper otherwise; 0 otherwise. The score of the
    latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
OPEN_L = -(L.SL_W + 0.024) / 2  # 62 mm round the 38 mm sleeper: 12 mm each side
GRIP_L = -(L.SL_W - 0.008) / 2  # 30 mm commanded across the sleeper
CLOSED = 0.0
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # both close along x

# the left arm: lift the sleeper's low end, hold it, lower it onto the shim
GRIP_Y = 0.155  # 15 mm from the sleeper's low end, 120 mm outboard of the low pad
H_START_GRIP = L.h_line(GRIP_Y, L.START_SLOPE, L.START_PIVOT)  # -0.94 mm: the underside at the grip as found
SL_TOP_AT_GRIP = L.LOW_H + H_START_GRIP + L.SL_H  # 171 mm
GRIP_TIPS_DOWN = 0.020  # pad tips 20 mm below the sleeper's top
GRIP_TCP = SL_TOP_AT_GRIP - GRIP_TIPS_DOWN + PAD_TIP
LIFT = 0.030  # commanded. GPU: the arm trails its command by 9.6 mm under the 4.7 N lift, so 20 mm gave only 10.4 mm,
# too little for the front edge of a shim pushed to level (7.5 mm thick) to clear the underside; 30 gives about 20 mm
LEVEL_TCP = GRIP_TCP + (L.DIP - H_START_GRIP)  # where the hand is when the sleeper rests level
PRESS = 0.005  # lowered 5 mm past that, so the sleeper rests its full weight on the shim before the fingers open
HOVER_L = 0.195
PARK_L = (0.22, 0.30, 0.200)

# the right arm: push the shim's butt with the closed fingers
PUSH_TIPS = L.LOW_H + 0.0025  # pad tips 2.5 mm over the low pad's top: on the butt face 2.5 to 9.5 mm up
PUSH_TCP = PUSH_TIPS + PAD_TIP
PUSH_START_X = L.SH_BUTT_X0 - PAD_FACE - 0.012  # the closed block comes down 12 mm short of the butt (in the air)
D_TARGET = 0.151  # tip travel past the front face: the middle of the 1 mm window (137 to 167 mm). GPU: 0.115 left
# the sleeper 2.4 mm low: it lies flush on the taper, so the shim's thickness at its REAR face sets its height
OVERTRAVEL = 0.0  # the shim slides on about 0.1 N with the end lifted: no give to make up


def push_end_x(d):
    return L.SH_BUTT_X0 + (d - (L.SH_TIP_X0 - L.SL_FRONT_X)) - PAD_FACE + OVERTRAVEL


HOVER_R = 0.150
PUSH_SPEED = 0.025  # m/s
BACK_OFF = 0.025
PARK_R = (0.14, -0.20, 0.150)


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def lift_moves():
    """The left arm: down round the sleeper's low end, grip it across its 38 mm, lift it 30 mm commanded (it pivots
    about the far pad). Ends holding it up."""
    x, y = L.SL_X, GRIP_Y
    return [
        _w(2.0, x, y, HOVER_L, OPEN_L, "over the sleeper's low end", YAW_L),
        _w(1.0, x, y, GRIP_TCP + 0.020, OPEN_L, "down to just above its top", YAW_L),
        _w(1.5, x, y, GRIP_TCP, OPEN_L, "slowly down round it", YAW_L),
        _w(0.5, x, y, GRIP_TCP, OPEN_L, "settle", YAW_L),
        _w(1.5, x, y, GRIP_TCP, GRIP_L, "grip the sleeper across its width", YAW_L),
        _w(2.0, x, y, GRIP_TCP + LIFT, GRIP_L, "lift its low end 30 mm (commanded) off the pad", YAW_L),
    ]


def lower_moves():
    x, y = L.SL_X, GRIP_Y
    return [
        _w(2.5, x, y, LEVEL_TCP - PRESS, GRIP_L, "lower it onto the shim", YAW_L),
        _w(0.5, x, y, LEVEL_TCP - PRESS, GRIP_L, "settle", YAW_L),
        _w(1.5, x, y, LEVEL_TCP - PRESS, OPEN_L, "let go", YAW_L),
        _w(1.5, x, y, HOVER_L, OPEN_L, "straight up", YAW_L),
        _w(1.5, *PARK_L[:2], PARK_L[2], OPEN, "aside", YAW_L),
    ]


def push_moves(d=D_TARGET, start_at=0.0):
    """The right arm: closed fingers in the air in front of the shim's butt, lowered to 2.5 mm over the pad's top,
    wait for the lift, push the butt until the tip is ``d`` past the sleeper's front face, back off 25 mm, up,
    aside."""
    y = L.SH_Y
    x1 = push_end_x(d)
    plan = [
        _w(2.0, PUSH_START_X, y, HOVER_R, OPEN, "over the air in front of the shim's butt", YAW_R),
        _w(1.5, PUSH_START_X, y, HOVER_R, CLOSED, "close the fingers into a block", YAW_R),
        _w(1.5, PUSH_START_X, y, PUSH_TCP + 0.030, CLOSED, "down to just above the pad's height", YAW_R),
        _w(1.5, PUSH_START_X, y, PUSH_TCP, CLOSED, "slowly down: pad tips 2.5 mm over the pad's top", YAW_R),
    ]
    waited = _duration(plan)
    if start_at > waited:
        plan.append(_w(start_at - waited, PUSH_START_X, y, PUSH_TCP, CLOSED, "wait for the lift", YAW_R))
    plan += [
        _w(abs(x1 - PUSH_START_X) / PUSH_SPEED, x1, y, PUSH_TCP, CLOSED, "push the shim in by its butt", YAW_R),
        _w(0.5, x1, y, PUSH_TCP, CLOSED, "settle", YAW_R),
        _w(1.0, x1 - BACK_OFF, y, PUSH_TCP, CLOSED, "back off 25 mm", YAW_R),
        _w(1.0, x1 - BACK_OFF, y, HOVER_R, OPEN, "straight up, opening", YAW_R),
        _w(1.5, *PARK_R[:2], PARK_R[2], OPEN, "aside", YAW_R),
    ]
    return plan


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


# Wrong ways (counterfactuals): where the shim's tip ends past the sleeper's front face.
VARIANT_D = {"short": 0.095, "overshoot": 0.200, "home": L.SH_L - 0.001}
VARIANTS = tuple(VARIANT_D)


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the left arm grips the sleeper's low end and lifts it; the right arm, already lowered in front of
    the shim's butt, pushes it in once the end is up, backs off; then the left arm lowers the sleeper onto the shim,
    lets go and moves aside."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    d = VARIANT_D.get(variant, D_TARGET)
    left = lift_moves()
    lifted = _duration(left)
    right = push_moves(d, start_at=lifted + 0.5)
    backed = _until(right, "back off")
    if backed > lifted:
        left.append(_w(backed - lifted, L.SL_X, GRIP_Y, GRIP_TCP + LIFT, GRIP_L, "hold it up while the shim goes in",
                       YAW_L))
    left += lower_moves()
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short      the shim pushed only 95 mm past the front face: 3.9 mm low end to end (0.25)
      overshoot  200 mm: 3.0 mm high (0.25)
      home       pushed fully home, the butt 1 mm short of the face: 3.1 mm high (0.25)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
