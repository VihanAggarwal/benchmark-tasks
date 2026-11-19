"""On the station table, real size (layout.py): a blue changing mat, an infant care manikin (newborn size, called
the doll in the code) on its back on it (head to the robot's left, its right arm on the robot's side lying along its
body), and a digital stick thermometer, switched off and display up, on the mat beside the doll's right hand. A
thin-film force sensor is taped on the armpit skin.

Goal: take the doll's temperature under its right arm the way the thermometer's manual says. Switch it on, lift the
arm, put the tip in the armpit touching the skin, lower the arm onto it and leave it for the 11 s peak, take it out,
lay it display up so the reading can be read, put the arm back at the doll's side and switch it off.

Mechanisms (parts.py): the doll is one loose mechkit articulation, its right arm one rigid link on the shoulder joint
(a revolute about the doll's left-right axis, joint friction 0.02 N m, no spring: let go below about 70 deg it sinks
back onto the doll's side or onto whatever is under it). The thermometer is a loose articulation with a sprung On/Off
button (0.8 mm, 2 N) and a display face that a declared rule (scoring.Unit, the manual's switching, self-test,
11 s peak and auto-off) slides under the window: blank when off, lit with Lo when on (``_display``, a mechkit
on_step callback). Everything else is contact. The arms are read through rigstate.py.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Trace: TASK_TRACE=<file> (one JSON line per
control step).
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as articulations)
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from offline_model import OfflineModel  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Take the infant care manikin's temperature under its right arm with the digital stick thermometer, "
               "leave the reading on show with the thermometer display up on the mat, put the arm back at its side "
               "and switch the thermometer off.")

OBJECTS = [P.MAT_SPEC, P.DOLL_SPEC, P.THERMO_SPEC]

_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineModel()


# -- the display rule (simulator) -------------------------------------------------------------------------------------
@mk.on_step
def _display(states) -> None:
    """Every control step in the simulator: the display face slides under the window while the unit is on (the
    device rule's state from the scorer's last step)."""
    mk.set_targets(P.THERMO_SPEC, "display", L.DISPLAY_TRAVEL if _TRACKER.unit.on else 0.0)


def _reset_all() -> None:
    _TRACKER.reset()
    OFFLINE.reset()


@rigstate.on_reset
def _new_episode() -> None:
    _reset_all()


@mk.on_reset
def _kit_reset() -> None:
    _reset_all()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the parts: mechkit has no environment AND the state handed in was not read
    from one (mechkit marks those ``source="sim"``)."""
    return mk.env() is None and getattr(objects.get(P.THERMO), "source", "sim") != "sim"


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only
            pass


def _mm(p, z0=L.TOP):
    return [round(float(p[0]) * 1000, 1), round(float(p[1]) * 1000, 1), round((float(p[2]) - z0) * 1000, 1)]


def success(objects) -> bool:
    """Runs first every control step: feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _reset_all()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    states = {n: objects.get(n) for n in P.PARTS}
    offline = no_simulator(objects)
    if offline:
        OFFLINE.step(arms, _TRACKER.unit.on)
        states = OFFLINE.states()
    result = _TRACKER.update(states, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    try:
        row["thermo_mm"] = _mm(states[P.THERMO].origin)
        row["thermo_yaw_deg"] = round(math.degrees(2 * math.atan2(float(states[P.THERMO].quat_xyzw[2]),
                                                                   float(states[P.THERMO].quat_xyzw[3]))), 1)
        row["display_mm"] = round(float(states[P.THERMO].joints["display"]) * 1000, 1)
        row["doll_mm"] = _mm(states[P.DOLL].origin)
    except Exception:  # noqa: BLE001 - never break scoring for a trace field
        pass
    if arms is not None:
        try:
            for side in rigstate.ARMS:
                row[f"tcp_{side}_mm"] = _mm(arms[side].tcp)
                row[f"open_{side}"] = round(float(arms[side].opening), 3)
        except Exception:  # noqa: BLE001
            row["rigstate_error"] = "unreadable snapshot"
    if offline:
        row["offline_events"] = [list(map(str, e)) for e in OFFLINE.events[-2:]]
    _trace(row)
    return result.success


def score(objects) -> float:
    """The score of the latest scored step (see scoring.py)."""
    return _TRACKER.last.score


# -- the scripted demo -------------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. World TCP points; the pad
# tips are 13 mm under the TCP; both grippers point straight down.
#   Right (thermometer): fingers closed, presses the On/Off button; picks the thermometer up across its body at its
#     centre of mass, carries it to the doll's right side and lays it on the mat 45 mm short of the armpit, along x;
#     re-grips it near its tail and, once the doll's arm is up, slides it along the mat until the tip touches the
#     armpit sensor; holds it while the arm comes down onto it; lets go. After the reading: slides it back out by its
#     tail; once the arm is back at the doll's side, re-grips it at its centre of mass and lays it back where it
#     started, display up; after 2 s presses the button again (off).
#   Left (doll's arm): pinches the doll's hand across x (the pinch's closing axis is parallel to the shoulder axis, so
#     the arm turns freely in it), lifts the arm in an arc to 40 deg and holds it (it sinks if let go); lowers it onto
#     the thermometer and lets go; after the reading lifts it off the thermometer to 40 deg, holds it while the right
#     slides the thermometer out, lowers it back to the doll's side and lets go.
# The left holds the arm from the robot's left and the right works low at the doll's side, so the two arms stay apart
# (the arm is held at 40 deg because higher the left wrist comes down over the right's slide).
TIP = 0.013
OPEN = -0.025  # 50 mm
GRIP_HAND = -0.009  # 18 mm commanded across the 26 mm hand
GRIP_TH = -0.007  # 14 mm commanded across the 22 mm body
CLOSED = -0.0005
YAW_HAND = -math.pi / 2  # left fingers close along x
YAW_TH = 0.0  # right fingers close along y
HAND_ARM, TH_ARM = "left", "right"
HAND_OVER = 0.006  # the TCP 6 mm over the hand's centre: pad tips 7 mm under it, 9 mm over the mat at rest (and the
# left gripper's housing 9 mm clear of the doll's diaper and legs)
LIFT_DEG = 40.0
OFF_DEG = 40.0
THETA_REST = P.support_angle(P.thermo_state(P.placed_origin()))
LOWER_TO = THETA_REST - math.radians(2.0)  # commanded just under where the arm rests on the thermometer
HOME = {"left": (0.200, 0.330, L.TOP + 0.200), "right": (0.200, -0.330, L.TOP + 0.200)}
MAT_TOP = L.TOP + L.MAT_H
TH_TCP = MAT_TOP + 0.002 + TIP  # pads' tips 2 mm over the mat beside the 13 mm body
TH_LIFT = TH_TCP + 0.045
LOW = TH_TCP + 0.010  # the right's height in and out at the doll's side: pad tips 12 mm over the mat
AWAY_LOW = (0.060, -0.040, LOW)  # the right's way in and out at the doll's side, under the left's reach
COM_X = float(P.THERMO_SPEC.total_center_of_mass()[0])  # 45 mm from the tail
TAIL_GRIP_X = 0.008  # the re-grip point near the tail, for sliding on the mat
BEHIND = 0.020  # the right's TCP this far behind the tail on its way in and out: the 27 mm wide pads 6.5 mm clear
SLIDE = 0.045
PUSH_PAST = 0.0025  # the slide is commanded 2.5 mm past the tip touching the film
START_TAIL = np.array(P.TH_ORIGIN[:2])
PLACE_TAIL = np.array(P.placed_origin()[:2])
PRE_TAIL = PLACE_TAIL - np.array([SLIDE, 0.0])
BUTTON_TOP = L.TH_H + L.BUTTON_PROUD
PRESS_TIPS = MAT_TOP + (BUTTON_TOP - 8.0) * L.MM  # 8 mm under the button's top (the arm gives under the push)


def _xyz(xy, z):
    return (float(xy[0]), float(xy[1]), float(z))


class _Plan:
    def __init__(self, yaw):
        self.wps: list = []
        self.t = 0.0
        self.yaw = yaw

    def add(self, dur, pos, finger, label):
        self.wps.append(Waypoint(dur, tuple(float(v) for v in pos), finger, label, self.yaw))
        self.t += dur
        return self.t

    @property
    def last(self):
        return self.wps[-1]

    def hold_until(self, t, label="hold"):
        if t > self.t + 1e-9:
            w = self.last
            self.add(t - self.t, w.pos, w.finger, label)
        return self.t


def _hand_tcp(theta):
    return P.hand_at(theta) + np.array([0.0, 0.0, HAND_OVER])


def _arc(hp: _Plan, th0: float, th1: float, finger, label, step_deg=5.0, per_step=0.5):
    n = max(1, int(math.ceil(abs(math.degrees(th1 - th0)) / step_deg)))
    for k in range(1, n + 1):
        th = th0 + (th1 - th0) * k / n
        hp.add(per_step, _hand_tcp(th), finger, f"{label} ({math.degrees(th):.0f} deg)")
    return hp.t


def _pinch(hp: _Plan, theta, label):
    at = _hand_tcp(theta)
    hp.add(2.5, at + (0.0, 0.0, 0.060), OPEN, f"over the doll's hand, open 50 mm ({label})")
    hp.add(1.5, at, OPEN, "down astride the hand")
    hp.add(0.3, at, OPEN, "settle")
    hp.add(1.5, at, GRIP_HAND, "pinch the hand (closing along the shoulder axis)")
    hp.add(0.3, at, GRIP_HAND, "settle the pinch")


PINCH_S = 2.5 + 1.5 + 0.3 + 1.5 + 0.3


def _release_hand(hp: _Plan):
    hp.add(1.5, hp.last.pos, OPEN, "let go of the hand")
    p = np.asarray(hp.last.pos)
    hp.add(1.2, (p[0], p[1], p[2] + 0.060), OPEN, "straight up")


def _press(tp: _Plan, tail_xy, label):
    b = np.asarray(tail_xy) + np.array([L.BUTTON_X * L.MM, 0.0])
    tp.add(2.0, _xyz(b, MAT_TOP + 0.090), CLOSED, f"over the On/Off button, fingers closed ({label})")
    tp.add(1.2, _xyz(b, PRESS_TIPS + TIP), CLOSED, f"press the button ({label})")
    tp.add(0.6, tp.last.pos, CLOSED, "hold the press")
    tp.add(1.0, _xyz(b, MAT_TOP + 0.090), CLOSED, "release: up off the button")


def _grip_th(tp: _Plan, tail_xy, at_x, label, above=True):
    g = np.asarray(tail_xy) + np.array([at_x, 0.0])
    if above:
        tp.add(1.8, _xyz(g, TH_LIFT + 0.015), OPEN, f"over the thermometer {label}, open 50 mm")
    tp.add(1.2, _xyz(g, TH_TCP), OPEN, "down astride its body")
    tp.add(0.3, tp.last.pos, OPEN, "settle")
    tp.add(1.5, tp.last.pos, GRIP_TH, f"grip it {label}")
    tp.add(0.3, tp.last.pos, GRIP_TH, "settle the grip")


def _let_go(tp: _Plan, label, up=True):
    tp.add(1.2, tp.last.pos, OPEN, f"let go ({label})")
    if up:
        p = np.asarray(tp.last.pos)
        tp.add(1.0, (p[0], p[1], TH_LIFT + 0.015), OPEN, "straight up")


def _carry(tp: _Plan, from_tail, to_tail, label):
    """Lift at the centre of mass, move, lay it down (it is gripped at COM_X from its tail)."""
    a = np.asarray(from_tail) + (COM_X, 0.0)
    b = np.asarray(to_tail) + (COM_X, 0.0)
    tp.add(1.2, _xyz(a, TH_LIFT), GRIP_TH, "lift it 45 mm")
    tp.add(3.0, _xyz(b, TH_LIFT), GRIP_TH, label)
    tp.add(1.5, _xyz(b, TH_TCP + 0.0005), GRIP_TH, "down onto the mat, display up")


def _leave(tp: _Plan, back=0.040):
    """The right's way out from the doll's side: back along x behind the tail (``back`` from where it let go), up a
    little (pad tips 1 mm under its top), out low toward the robot's right, home."""
    p = np.asarray(tp.last.pos)
    tp.add(1.0, (p[0] - back, p[1], p[2]), OPEN, "back along x, behind its tail")
    tp.add(0.5, (p[0] - back, p[1], LOW), OPEN, "up a little")
    tp.add(1.5, AWAY_LOW, OPEN, "out, low, toward the robot's right")
    tp.add(1.5, HOME[TH_ARM], OPEN, "home")


def _come(tp: _Plan, tail_xy, at_x, label):
    """The right's way in: low from the robot's right to behind the tail, down, forward astride the body, grip."""
    behind = np.asarray(tail_xy) + (-BEHIND, 0.0)
    tp.add(2.0, AWAY_LOW, OPEN, "in, low, from the robot's right")
    tp.add(1.5, _xyz(behind, LOW), OPEN, "behind the thermometer's tail, open 50 mm")
    tp.add(0.5, _xyz(behind, TH_TCP), OPEN, "down")
    tp.add(1.5, _xyz(np.asarray(tail_xy) + (at_x, 0.0), TH_TCP), OPEN, f"forward astride its body ({label})")
    _grip_th(tp, tail_xy, at_x, label, above=False)


def plans() -> dict:
    tp, hp = _Plan(YAW_TH), _Plan(YAW_HAND)
    # A, right: switch on; pick it up at its centre of mass; lay it on the mat at the doll's right side, 45 mm short of
    # the armpit; out of the way.
    _press(tp, START_TAIL, "on")
    _grip_th(tp, START_TAIL, COM_X, "at its centre of mass")
    _carry(tp, START_TAIL, PRE_TAIL, "to the doll's right side, 45 mm short of the armpit")
    _let_go(tp, "on the mat", up=False)
    # let go at its centre of mass: back COM_X + BEHIND, so the open fingers are behind its tail before they go out
    # sideways (GPU 19 Nov: backed only 40 mm, the left-hand finger swept its tail 17 mm and turned it 12 deg)
    _leave(tp, back=COM_X + BEHIND)
    # B, left: pinch the hand and lift the arm in an arc to 40 deg (the right out of the way).
    hp.add(2.0, HOME[HAND_ARM], OPEN, "wait at home")
    hp.hold_until(tp.t - 1.5, "wait at home while the thermometer is carried")
    _pinch(hp, 0.0, "at the doll's side")
    _arc(hp, 0.0, math.radians(LIFT_DEG), GRIP_HAND, "lift the arm up off its side")
    # C, right: grip it near its tail, slide it in along the mat until the tip touches the armpit sensor, let go,
    # out of the way (the thermometer stays where it is put).
    tp.hold_until(hp.t, "wait at home for the arm to be up")
    _come(tp, PRE_TAIL, TAIL_GRIP_X, "near its tail")
    slide_to = PLACE_TAIL + (TAIL_GRIP_X + PUSH_PAST, 0.0)
    tp.add(2.5, _xyz(slide_to, TH_TCP), GRIP_TH, "slide it in under the arm: tip onto the armpit")
    tp.add(0.5, tp.last.pos, GRIP_TH, "hold it there")
    _let_go(tp, "tip on the armpit", up=False)
    _leave(tp)
    # D, left: once the right is out of the way, lower the arm onto it, let go of the hand, up, home.
    hp.hold_until(tp.t - 1.5, "hold the arm up")
    _arc(hp, math.radians(LIFT_DEG), LOWER_TO, GRIP_HAND, "lower the arm onto the thermometer")
    t_contact = hp.t - 0.5  # the arm lands on it on the last step of the arc
    hp.add(0.5, hp.last.pos, GRIP_HAND, "settle")
    _release_hand(hp)
    hp.add(2.0, HOME[HAND_ARM], OPEN, "home")
    # E, left: after the reading, pinch the hand where the arm rests and lift it off the thermometer.
    # the open fingers come down round the hand only once the 11 s count is over (with 1 s to spare)
    hp.hold_until(t_contact + L.PEAK_S + 4.0 - PINCH_S, "wait for the reading (11 s)")
    _pinch(hp, THETA_REST, "resting on the thermometer")
    _arc(hp, THETA_REST, math.radians(OFF_DEG), GRIP_HAND, "lift the arm off the thermometer")
    # F, right: behind its tail, grip, slide it out of the armpit, let go, out of the way.
    tp.hold_until(hp.t, "wait at home for the arm to be up")
    _come(tp, PLACE_TAIL, TAIL_GRIP_X, "near its tail")
    tp.add(2.0, _xyz(PRE_TAIL + (TAIL_GRIP_X, 0.0), TH_TCP), GRIP_TH, "slide it out of the armpit")
    _let_go(tp, "out of the armpit", up=False)
    _leave(tp, back=0.028)
    # G, left: lower the arm back to the doll's side, let go, home.
    hp.hold_until(tp.t - 1.5, "hold the arm up")
    _arc(hp, math.radians(OFF_DEG), 0.0, GRIP_HAND, "lower the arm back to the doll's side")
    hp.add(0.5, hp.last.pos, GRIP_HAND, "settle")
    _release_hand(hp)
    hp.add(2.0, HOME[HAND_ARM], OPEN, "home")
    # H, right: re-grip it at its centre of mass, lay it back where it started, display up; after 2 s with both
    # hands off, switch it off; home.
    tp.hold_until(hp.t, "wait at home for the left to be home")
    _come(tp, PRE_TAIL, COM_X, "at its centre of mass")
    _carry(tp, PRE_TAIL, START_TAIL, "back to where it started")
    _let_go(tp, "on the mat, display up")
    p = np.asarray(tp.last.pos)
    tp.add(1.0, (p[0], p[1], MAT_TOP + 0.120), OPEN, "up, well clear of it")
    tp.hold_until(max(tp.t, hp.t) + 2.8, "leave it shown")
    _press(tp, START_TAIL, "off")
    tp.add(2.0, HOME[TH_ARM], OPEN, "home")
    end = max(tp.t, hp.t)
    tp.hold_until(end, "wait")
    hp.hold_until(end, "wait")
    return {TH_ARM: tp.wps, HAND_ARM: hp.wps}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
