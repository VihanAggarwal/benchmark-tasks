"""On the station table, real size: a Braun MultiQuick 5 Vario hand blender (MQ5235 class), unplugged, lies on its back
along y at x 300 mm, its coupling end at y 20 mm facing -y, its blending shaft clicked into that end. Its motor part
(228 mm, 64 x 60 mm, 0.62 kg) has a square socket (41 mm) in the coupling end and, 45 mm back from it, one release
button on each flank (14 x 22 mm, standing 3 mm proud, 3 mm travel against a 2 N spring, est). Each button is one link
with a latch hook that springs in behind the lip of the attachment's stub. A wire whisk (gearbox 58 mm, 0.11 kg)
lies at x 200 mm, its gearbox's stub toward +y. A 200 x 200 x 10 mm cork mat is fixed at x 440 mm, y -170 mm.

Goal: the shaft off the motor part and on the mat, the whisk pushed home with both hooks behind its lip, hands off.
The shaft only comes off with both buttons in (one hook still in binds it); the whisk goes on past both hooks with a
click.

The parts are mechkit compounds (mechkit.py, next to this file): the motor part is one loose articulation (body and
two sprung slider links), the shaft and the whisk are loose compounds, the mat a fixed one. There is no declared
software rule: the latches are geometry (a hook wedge and the lip's flat faces) and springs. The arms are read through
rigstate.py (the let-go check).

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
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import BTN_N, BTN_P, MOTOR, SHAFT, WHISK  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Take the blending shaft off the hand blender and lay it on the cork mat, then fit the whisk to the "
               "blender and let go of everything.")


# -- the parts --------------------------------------------------------------------------------------------------------
def _button(name: str, axis: str, s: int):
    return mk.slider(name, axis, P.button_shapes(s), upper=L.HOOK_OUT, lower=-L.BTN_TRAVEL, mass=L.BTN_MASS,
                     spring=L.BTN_SPRING, damping=L.BTN_DAMP, friction=0.0, color=L.BTN_BLACK)


MOTOR_PART = mk.assembly(MOTOR, origin=L.M_ORIGIN, mass=L.M_BASE_MASS, shapes=P.motor_shapes(),
                         joints=[_button(BTN_P, "X", +1), _button(BTN_N, "-X", -1)], color=L.M_WHITE,
                         friction=L.M_FRICTION, tight_fit=True)
SHAFT_PART = mk.assembly(SHAFT, origin=L.SH_ORIGIN, mass=L.SH_MASS, shapes=P.shaft_shapes(), color=L.SH_STEEL,
                         friction=L.SH_FRICTION, tight_fit=True)
WHISK_PART = mk.assembly(WHISK, origin=L.WK_ORIGIN, mass=L.WK_MASS, shapes=P.whisk_shapes(), color=L.WK_GREY,
                         friction=L.WK_FRICTION, tight_fit=True)
MAT_PART = mk.assembly("mat", origin=(L.MAT_X, L.MAT_Y, L.TOP), shapes=P.mat_shapes(), color=L.MAT_CORK,
                       friction=L.MAT_FRICTION)
OBJECTS = [MAT_PART, MOTOR_PART, SHAFT_PART, WHISK_PART]
SPECS = {MOTOR: MOTOR_PART, SHAFT: SHAFT_PART, WHISK: WHISK_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it has no sockets, buttons or hooks.
# Offline only (never in the simulator), this model gives the scorer: (1) the buttons pushed in by a closed gripper
# straddling the motor part at the buttons; (2) the motor part carried by a gripper closed on it; (3) the shaft or the
# whisk carried by a gripper closed on its bell or gearbox, held in the socket by a hook that is still in (the pair is
# dragged along if the motor part is not held), sliding only along the axis while its stub is inside; (4) a part let
# go outside the socket dropping to the mat or the table, one let go inside staying put.
PAD_HALF = 0.0135  # a pad's half width along y (rigstate.FINGER_HALF_WIDTH)
SIDE_TOL = 0.012  # the pads straddle a part: its axis within 12 mm of the TCP across the closing direction
CAPTURE_LAT = 0.0065  # a stub starts into the socket if its axis is within 1.5 mm plus the 5 mm funnel of it
BORE_PLAY = 0.0015


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.org = {MOTOR: np.array(L.M_ORIGIN, dtype=float), SHAFT: np.array(L.SH_ORIGIN, dtype=float),
                    WHISK: np.array(L.WK_ORIGIN, dtype=float)}
        self.q = {BTN_P: 0.0, BTN_N: 0.0}
        self.hold: dict = {}  # arm side -> (part, offset from TCP)

    # geometry helpers --------------------------------------------------------------------------------------------
    def rel(self, name):
        return self.org[name] - self.org[MOTOR]

    def inserted(self, name) -> bool:
        r = self.rel(name)
        return bool(math.hypot(r[0], r[2]) <= 0.004 and r[1] + L.STUB_LEN > L.INSERT_MIN)

    def hooks_in(self) -> bool:
        return any(self.q[j] >= -L.HOOK_IN_MAX for j in (BTN_P, BTN_N))

    def hooks_clear(self) -> bool:
        return all(self.q[j] <= -L.HOOK_CLEARS for j in (BTN_P, BTN_N))

    def engaged(self, name) -> bool:
        """The attachment's lip is behind a hook that is in: it cannot come out."""
        return bool(self.inserted(name) and self.rel(name)[1] >= -L.HOOK_IN_MAX and self.hooks_in())

    @staticmethod
    def _width(name) -> float:
        return {SHAFT: 2 * L.SH_BELL_R, WHISK: 2 * L.WK_BOX_R, MOTOR: L.M_W}[name]

    def _near(self, name, tcp, tips) -> bool:
        o = self.org[name]
        z_lo, z_hi = float(tips[2]), float(tips[2]) + rigstate.FINGER_LEN
        if name == SHAFT:
            ok_y = o[1] + L.SH_BELL_U0 <= tcp[1] <= o[1] + L.SH_STEM_U1
            return bool(abs(tcp[0] - o[0]) <= SIDE_TOL and ok_y and z_lo <= o[2] <= z_hi)
        if name == WHISK:
            ok_y = o[1] - L.WK_BOX_LEN <= tcp[1] <= o[1]
            return bool(abs(tcp[0] - o[0]) <= SIDE_TOL and ok_y and z_lo <= o[2] <= z_hi)
        ok_y = o[1] <= tcp[1] <= o[1] + L.M_LEN
        return bool(abs(tcp[0] - o[0]) <= SIDE_TOL and ok_y and z_lo <= o[2] + L.M_H / 2 and z_hi >= o[2] - L.M_H / 2)

    def _presses_buttons(self, tcp, tips, gap) -> bool:
        o = self.org[MOTOR]
        by = o[1] + L.BTN_Y
        z_lo, z_hi = float(tips[2]), float(tips[2]) + rigstate.FINGER_LEN
        return bool(abs(tcp[0] - o[0]) <= SIDE_TOL and abs(tcp[1] - by) <= PAD_HALF + L.BTN_Y_LEN / 2 - 0.004
                    and z_lo <= o[2] + L.BTN_Z_LEN / 2 and z_hi >= o[2] - L.BTN_Z_LEN / 2
                    and gap <= L.M_W + 2 * L.BTN_PROUD - 2 * L.HOOK_CLEARS)

    # one control step ---------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        pressed = False
        current = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = opening * rigstate.FULL_GAP
            current[side] = (tcp, tips, gap)
            if self._presses_buttons(tcp, tips, gap):
                pressed = True
        depth = 0.0
        if pressed:
            gaps = [g for (_, _, g) in current.values() if g <= L.M_W + 2 * L.BTN_PROUD]
            depth = min(L.BTN_TRAVEL, max(0.0, (L.M_W + 2 * L.BTN_PROUD - min(gaps)) / 2))
        self.q = {BTN_P: -depth, BTN_N: -depth}
        # grips made and let go
        for side, (tcp, tips, gap) in current.items():
            if side in self.hold:
                name, off = self.hold[side]
                if gap > self._width(name) + 0.004:
                    del self.hold[side]
                continue
            for name in (SHAFT, WHISK, MOTOR):
                if gap <= self._width(name) + 0.002 and self._near(name, tcp, tips):
                    self.hold[side] = (name, self.org[name] - tcp)
                    break
        # carry what is held (the motor part first: a held motor part is what the others pull against)
        held = {name: (side, off) for side, (name, off) in self.hold.items()}
        if MOTOR in held:
            side, off = held[MOTOR]
            self.org[MOTOR] = current[side][0] + off
        for name in (SHAFT, WHISK):
            if name not in held:
                continue
            side, off = held[name]
            self._carry(name, current[side][0] + off, MOTOR in held)
        for name in (SHAFT, WHISK):
            if name not in held:
                self._drop(name)

    def _carry(self, name, want, motor_held) -> None:
        o, m = self.org[name], self.org[MOTOR]
        if self.inserted(name):
            if self.engaged(name):
                if not motor_held:  # the hook drags the motor part along
                    delta = want - o
                    self.org[MOTOR] = m + np.array([delta[0], delta[1], 0.0])
                    self.org[name] = o + np.array([delta[0], delta[1], 0.0])
                    for n in (SHAFT, WHISK):
                        if n != name and self.inserted(n):
                            self.org[n] = self.org[n] + np.array([delta[0], delta[1], 0.0])
                return  # held motor part: the arm stalls, nothing moves
            y = min(float(want[1]), float(m[1]))  # the shoulder stops at the coupling face
            if name == WHISK and self.engaged(name):
                y = max(y, float(m[1]) - L.HOOK_IN_MAX)
            self.org[name] = np.array([m[0] + float(np.clip(want[0] - m[0], -BORE_PLAY, BORE_PLAY)), y,
                                       m[2] + float(np.clip(want[2] - m[2], -BORE_PLAY, BORE_PLAY))])
            return
        lat = math.hypot(float(want[0] - m[0]), float(want[2] - m[2]))
        tip = float(want[1] - m[1]) + L.STUB_LEN
        if tip > 0.0 and name in (SHAFT, WHISK) and lat > CAPTURE_LAT:
            want = np.array([want[0], m[1] - L.STUB_LEN, want[2]])  # the stub's end on the face: it cannot go in
        elif tip > 0.0 and lat <= CAPTURE_LAT:
            y = min(float(want[1]), float(m[1]))
            want = np.array([m[0] + float(np.clip(want[0] - m[0], -BORE_PLAY, BORE_PLAY)), y,
                             m[2] + float(np.clip(want[2] - m[2], -BORE_PLAY, BORE_PLAY))])
        self.org[name] = np.asarray(want, dtype=float)

    def _drop(self, name) -> None:
        if self.inserted(name):
            return
        o = self.org[name]
        if name == SHAFT:
            c = o + np.array([0.0, S.SHAFT_MID_U, 0.0])
            over = abs(c[0] - L.MAT_X) <= L.MAT_HALF and abs(c[1] - L.MAT_Y) <= L.MAT_HALF
            self.org[name] = np.array([o[0], o[1], L.TOP + (L.MAT_T if over else 0.0) + L.SH_BELL_R])
        else:
            self.org[name] = np.array([o[0], o[1], L.TOP + L.WK_BOX_R])

    def states(self) -> dict:
        q = (0.0, 0.0, 0.0, 1.0)
        return {MOTOR: mk.state_from_poses(MOTOR_PART, (*self.org[MOTOR], *q), joints=dict(self.q), source="offline"),
                SHAFT: mk.state_from_poses(SHAFT_PART, (*self.org[SHAFT], *q), source="offline"),
                WHISK: mk.state_from_poses(WHISK_PART, (*self.org[WHISK], *q), source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(MOTOR), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, btn_p mm, btn_n mm, shaft p mm, whisk p mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: every part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The shaft off the motor part and on the mat, the whisk home on both hooks, both grippers 30 mm clear and every
    part still for 1 s, no latched fall; held 1 s. This runs first every control step, so it also feeds the tracker
    (and, without a simulator, the offline stand-in)."""
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
    row["source"] = getattr(parts.get(MOTOR), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["btn_p_mm"], d["btn_n_mm"], d["shaft_p_mm"], d["whisk_p_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a part fell off the table, or unless every part is let go and still, or the shaft is still on; 1.0 the shaft
    on the mat and the whisk home with both hooks behind its lip; 0.5 the whisk on with the shaft elsewhere, or the
    shaft on the mat and the whisk home on one hook; 0.25 the shaft off otherwise. The score of the latest scored
    step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
OPEN = -0.04695  # 94 mm
GRIP_MOTOR = -(L.M_W - 0.008) / 2  # 56 mm across the 64 mm body: the pads press both buttons flush
GRIP_BELL = -(2 * L.SH_BELL_R - 0.008) / 2  # 52 mm on the 60 mm bell
GRIP_GEAR = -(2 * L.WK_BOX_R - 0.008) / 2  # 50 mm on the 58 mm gearbox
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # both close along x

Z_GRIP = 0.037  # TCP height: pad tips 24 mm over the table, so the finger carriages clear the body's 60 mm top
HOVER = 0.110
Y_BTN = L.M_Y0 + L.BTN_Y  # 45 mm: the buttons
Y_HOLD = 0.190  # where the left hand re-grips the motor part, 170 mm back from the coupling face
BELL_GRIP_U = -0.1555  # the right hand's grip on the shaft: round the bell's rear end, 41 mm from its centre of mass,
# so that the two hands stay 200 mm apart (the arms' forearms need that)
GEAR_GRIP_U = -0.045  # and on the gearbox: 15 mm from its rear end
PULL = 0.040  # the shaft is drawn 40 mm out (its stub is 21 mm long)
Z_CARRY = 0.085  # a carried shaft's TCP height
PUSH_SPEED = 0.030  # m/s
OVERSHOOT = 0.008  # commanded past the stop: about 3 N of the arm's give
PARK_L = (0.30, 0.36, HOVER)
PARK_R = (0.16, -0.10, HOVER)
PARK_R_NO_WHISK = (0.41, -0.40, HOVER)  # the whisk is still lying at x 200 mm
SHAFT_AT = {"": (L.MAT_X, L.MAT_Y), "shaft_table": (0.41, -0.40)}  # where the shaft's centre goes


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
    """The right arm: grip the shaft's bell, draw it out (the left hand holds both buttons in), carry it to the mat
    and let it down, then fetch the whisk by its gearbox, carry it in line with the socket and push it home."""
    x0, y0 = L.M_X, L.M_Y0 + BELL_GRIP_U
    cx, cy = SHAFT_AT.get(variant, SHAFT_AT[""])
    # shaft origin y so that the shaft's centre sits at cy; the TCP is BELL_GRIP_U behind the origin
    y_put = cy - S.SHAFT_MID_U + BELL_GRIP_U
    z_put = (0.0 if variant == "shaft_table" else L.MAT_T) + L.SH_BELL_R + (Z_GRIP - L.SH_AXIS_Z) + 0.0005
    plan = [
        _w(2.5, x0, y0, HOVER, OPEN, "over the bell", YAW_R),
        _w(1.2, x0, y0, Z_GRIP + 0.020, OPEN, "down to just above the bell", YAW_R),
        _w(1.0, x0, y0, Z_GRIP, OPEN, "slowly down beside it", YAW_R),
        _w(0.3, x0, y0, Z_GRIP, OPEN, "settle", YAW_R),
        _w(1.5, x0, y0, Z_GRIP, GRIP_BELL, "grip the bell", YAW_R),
        _w(0.5, x0, y0, Z_GRIP, GRIP_BELL, "wait for the buttons", YAW_R),
        _w(2.0, x0, y0 - PULL, Z_GRIP, GRIP_BELL, "draw the shaft out along its axis", YAW_R),
        _w(1.0, x0, y0 - PULL, Z_CARRY, GRIP_BELL, "lift it clear", YAW_R),
        _w(2.5, cx, y_put, Z_CARRY, GRIP_BELL, "carry it over the mat", YAW_R),
        _w(1.5, cx, y_put, z_put, GRIP_BELL, "lower it onto the mat", YAW_R),
        _w(1.5, cx, y_put, z_put, OPEN, "let go of it", YAW_R),
        _w(1.0, cx, y_put, HOVER, OPEN, "straight up", YAW_R),
    ]
    if variant == "no_whisk":
        return plan + [_w(1.5, *PARK_R_NO_WHISK, OPEN, "aside", YAW_R)]
    wy = L.WK_SHOULDER_Y + GEAR_GRIP_U  # the TCP on the gearbox as found
    ay = L.M_Y0 - 0.085 + GEAR_GRIP_U  # the shoulder 85 mm short of the coupling face: the stub's end 64 mm short
    y_home = L.M_Y0 + GEAR_GRIP_U  # the shoulder on the face
    z_line = L.M_AXIS_Z + (Z_GRIP - L.WK_AXIS_Z)  # the whisk's axis level with the socket's
    plan += [
        _w(2.0, L.WK_X, wy, HOVER, OPEN, "over the gearbox", YAW_R),
        _w(1.2, L.WK_X, wy, Z_GRIP + 0.020, OPEN, "down to just above it", YAW_R),
        _w(1.0, L.WK_X, wy, Z_GRIP, OPEN, "slowly down beside it", YAW_R),
        _w(0.3, L.WK_X, wy, Z_GRIP, OPEN, "settle", YAW_R),
        _w(1.5, L.WK_X, wy, Z_GRIP, GRIP_GEAR, "grip the gearbox", YAW_R),
        _w(1.0, L.WK_X, wy, Z_GRIP + 0.025, GRIP_GEAR, "lift the whisk 25 mm", YAW_R),
        _w(2.0, L.M_X, ay, Z_GRIP + 0.025, GRIP_GEAR, "carry it in line with the socket", YAW_R),
        _w(1.0, L.M_X, ay, z_line, GRIP_GEAR, "lower it to the socket's axis", YAW_R),
        _w((y_home + OVERSHOOT - ay) / PUSH_SPEED, L.M_X, y_home + OVERSHOOT, z_line, GRIP_GEAR,
           "push the stub in past both hooks until the shoulder is on the face", YAW_R),
        _w(0.5, L.M_X, y_home + OVERSHOOT, z_line, GRIP_GEAR, "settle", YAW_R),
        _w(1.5, L.M_X, y_home + OVERSHOOT, z_line, OPEN, "let go of the whisk", YAW_R),
        _w(1.0, L.M_X, y_home + OVERSHOOT, HOVER, OPEN, "straight up", YAW_R),
        _w(1.5, *PARK_R, OPEN, "aside", YAW_R),
    ]
    return plan


def left_plan(variant: str, right: list) -> list:
    """The left arm: squeeze both buttons in with the pads closed across the body, hold the motor part while the shaft
    is drawn out, let go, re-grip the body away from the buttons, hold it while the whisk is pushed in, let go."""
    x = L.M_X
    y1 = Y_HOLD if variant == "grip_away" else Y_BTN
    plan = [
        _w(2.5, x, y1, HOVER, OPEN, "over the buttons" if y1 == Y_BTN else "over the body", YAW_L),
        _w(1.2, x, y1, Z_GRIP + 0.020, OPEN, "down to just above the body", YAW_L),
        _w(1.0, x, y1, Z_GRIP, OPEN, "slowly down beside the body", YAW_L),
        _w(0.3, x, y1, Z_GRIP, OPEN, "settle", YAW_L),
        _w(1.5, x, y1, Z_GRIP, GRIP_MOTOR, "squeeze: both buttons in" if y1 == Y_BTN else "grip the body", YAW_L),
    ]
    drawn = _until(right, "draw the shaft out")
    plan.append(_w(drawn - _duration(plan), x, y1, Z_GRIP, GRIP_MOTOR, "hold while the shaft is drawn out", YAW_L))
    plan += [
        _w(1.5, x, y1, Z_GRIP, OPEN, "let go: the buttons spring out", YAW_L),
        _w(1.0, x, y1, HOVER, OPEN, "straight up", YAW_L),
    ]
    if variant == "no_whisk":
        return plan + [_w(1.5, *PARK_L, OPEN, "aside", YAW_L)]
    plan += [
        _w(1.5, x, Y_HOLD, HOVER, OPEN, "over the body, away from the buttons", YAW_L),
        _w(1.2, x, Y_HOLD, Z_GRIP + 0.020, OPEN, "down to just above the body", YAW_L),
        _w(1.0, x, Y_HOLD, Z_GRIP, OPEN, "slowly down beside the body", YAW_L),
        _w(0.3, x, Y_HOLD, Z_GRIP, OPEN, "settle", YAW_L),
        _w(1.5, x, Y_HOLD, Z_GRIP, GRIP_MOTOR, "grip the body", YAW_L),
    ]
    pushed = _push_settled(right)
    plan.append(_w(max(0.0, pushed - _duration(plan)), x, Y_HOLD, Z_GRIP, GRIP_MOTOR, "hold while the whisk goes in",
                   YAW_L))
    plan += [
        _w(1.5, x, Y_HOLD, Z_GRIP, OPEN, "let go", YAW_L),
        _w(1.0, x, Y_HOLD, HOVER, OPEN, "straight up", YAW_L),
        _w(1.5, *PARK_L, OPEN, "aside", YAW_L),
    ]
    return plan


def _push_settled(right) -> float:
    """When the right arm's push is over (its 0.5 s settle after the push)."""
    t = 0.0
    seen = False
    for w in right:
        t += w.duration
        if w.label.startswith("push the stub in"):
            seen = True
        elif seen and w.label == "settle":
            return t
    raise KeyError("push")


# Wrong ways (counterfactuals), each with the score it earns.
VARIANTS = ("shaft_table", "no_whisk", "grip_away")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: both hands go down together: the left closes across the motor part's buttons, the right on the
    shaft's bell; the right draws the shaft out and carries it to the mat while the left lets go and re-grips the body;
    the right fetches the whisk and pushes it onto the socket while the left holds the body; both let go."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = right_plan(variant)
    return {"left": left_plan(variant, right), "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      shaft_table  the shaft laid on the table 230 mm off the mat, the whisk fitted: 0.5
      no_whisk     the shaft on the mat, the whisk left lying: 0.25
      grip_away    the left hand grips the body away from the buttons: both hooks hold the shaft: 0.0"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
