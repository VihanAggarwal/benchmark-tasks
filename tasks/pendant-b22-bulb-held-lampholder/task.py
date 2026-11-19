"""On the station table, real size: a rigid stand standing in for the ceiling (a 30 mm square steel post at x 450 mm
with a top arm, underside 500 mm up, reaching back over x 300 mm, y 0). A plain white pendant set hangs from the arm:
a ceiling rose, 240 mm of round flex and a B22d lampholder, mouth down, 178 mm over the table. A new A60 B22 bulb
(60 mm across, 104 mm long, 40 g) stands cap-up in a 50 mm ring stand at x 250 mm, y -130 mm.

Goal: the bulb locked in the holder (pushed up against the two plungers, turned 15 degrees, dropped into the hooks)
and both hands off, the holder hanging where it hung. The holder hangs on a soft flex: pushed from below it rides up,
turned it spins. So one hand holds the holder steady and the other pushes and turns the bulb.

The pendant is one fixed-base mechkit articulation (mechkit.py, next to this file): the rose; the flex as two swing
hinges at the rose, a free slack slide up the flex (it buckles when the holder is pushed up), a stretch spring down it
(pulled tight against the rose) and the holder's weak twist hinge; in the holder two sprung plungers and, in each
J-slot, a catch (the hook's floor, with two lips) on a short slide. The J's shape is declared: a rule
(scoring.Bayonet, run every control step through ``mk.on_step``) reads the bulb in the holder's frame and drives the
catches in under the pins when the pins have gone up the leg, turned at the top and are over the hooks, and out again
when they leave. The plungers, the catches' floors and lips, the bore, the bulb and every push and turn are physical
contact. The arms are read through rigstate.py (the hands-off check). tiltgrip.py wraps
``yam_arena.demo.grip_rotation`` so that the demo's waypoints can carry pitched grips.

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
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (pitched grips for the demo: wraps yam_arena.demo.grip_rotation)
from scoring import BULB, PENDANT, RING, STAND  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of hands-off and hanging

INSTRUCTION = "Fit the new bulb into the hanging pendant lampholder so it locks in."


# -- the parts --------------------------------------------------------------------------------------------------------
_SWING_AT = (0.0, 0.0, L.SWING_Z)
SWING_X = mk.hinge("swing_x", "X", _SWING_AT, P.swing_hub_shapes(), lower=-L.SWING_LIMIT, upper=L.SWING_LIMIT,
                   mass=L.LINK_MASS, damping=L.SWING_DAMP, color=L.FLEX_WHITE)
SWING_Y = mk.hinge("swing_y", "Y", _SWING_AT, P.flex_shapes(), lower=-L.SWING_LIMIT, upper=L.SWING_LIMIT,
                   mass=L.FLEX_MASS, damping=L.SWING_DAMP, parent="swing_x", color=L.FLEX_WHITE)
SLACK = mk.slider("slack", "Z", P.slack_shapes(), upper=L.SLACK_UP, anchor=(0.0, 0.0, P.SLACK_Z[0]),
                  mass=L.LINK_MASS, damping=L.SLACK_DAMP, parent="swing_y", color=L.FLEX_WHITE)
STRETCH = mk.slider("stretch", "-Z", P.cord_grip_shapes(), upper=L.STRETCH_MAX, anchor=(0.0, 0.0, L.HOLDER_TOP),
                    mass=L.LINK_MASS, spring=L.STRETCH_K, damping=L.STRETCH_DAMP, parent="slack", color=L.WHITE)
HOLDER = mk.hinge("holder", "Z", (0.0, 0.0, L.MOUTH_Z), P.holder_shapes(), lower=-L.TWIST_LIMIT,
                  upper=L.TWIST_LIMIT, mass=L.HOLDER_MASS, spring=L.TWIST_K, damping=L.TWIST_DAMP, parent="stretch",
                  color=L.WHITE)
PLUNGERS = tuple(mk.slider(f"plunger_{n}", "Z", P.plunger_shapes(s), upper=L.PL_TRAVEL, mass=L.LINK_MASS,
                           spring=L.PL_K, damping=L.PL_DAMP, parent="holder", color=L.CAP_METAL)
                 for n, s in (("a", 1), ("b", -1)))
CATCH_LINKS = (mk.slider("catch_a", "-X", P.catch_shapes(1), upper=L.CATCH_TRAVEL, mass=L.LINK_MASS,
                         spring=L.CATCH_K, damping=L.CATCH_DAMP, parent="holder", color=L.WHITE),
               mk.slider("catch_b", "X", P.catch_shapes(-1), upper=L.CATCH_TRAVEL, mass=L.LINK_MASS,
                         spring=L.CATCH_K, damping=L.CATCH_DAMP, parent="holder", color=L.WHITE))
PENDANT_PART = mk.assembly(PENDANT, origin=(L.PEND_X, L.PEND_Y, L.TOP), shapes=P.rose_shapes(),
                           joints=[SWING_X, SWING_Y, SLACK, STRETCH, HOLDER, *PLUNGERS, *CATCH_LINKS], color=L.WHITE,
                           friction=L.HOLDER_FRICTION, tight_fit=True)
STAND_PART = mk.assembly(STAND, origin=(L.POST_X, L.POST_Y, L.TOP), shapes=P.stand_shapes(), color=L.STEEL)
RING_PART = mk.assembly(RING, origin=(L.RING_X, L.RING_Y, L.TOP), shapes=P.ring_stand_shapes(), color=L.STAND_GREY)
BULB_PART = mk.assembly(BULB, origin=(L.RING_X, L.RING_Y, L.TOP + L.BULB_START_Z), shapes=P.bulb_shapes(),
                        mass=L.BULB_MASS, yaw=math.radians(L.BULB_START_YAW_DEG), color=L.FROSTED,
                        friction=L.BULB_FRICTION, tight_fit=True)
OBJECTS = [STAND_PART, PENDANT_PART, RING_PART, BULB_PART]
SPECS = {STAND: STAND_PART, PENDANT: PENDANT_PART, RING: RING_PART, BULB: BULB_PART}
FIXED = {STAND: STAND_PART, RING: RING_PART}


# -- the J-slot rule in the simulator ------------------------------------------------------------------------------
RULE = S.Bayonet()


def catch_target(hooked: bool) -> float:
    return L.CATCH_IN if hooked else L.CATCH_OUT


@mk.on_step
def _bayonet(states) -> None:
    """Every control step (sim): read the bulb in the holder, run the J rule, drive the catches; keep the plungers'
    preload (targets go back to 0 at every reset, so they are set every step)."""
    r = S.read(states)
    if r is not None:
        RULE.update(r.lateral, r.h_end, r.tilt_deg, r.turn_deg)
    for name in S.CATCHES:
        mk.set_targets(PENDANT_PART, name, catch_target(RULE.hooked))
    for j in PLUNGERS:
        mk.set_targets(PENDANT_PART, j.name, -L.PL_PRELOAD)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows no flex, plungers or hooks. Offline only (never in the simulator), this model gives the
# scorer: (1) the holder held still by a gripper closed under 42 mm with its TCP within 8 mm of the holder's axis,
# 10 to 52 mm over the mouth; (2) the bulb carried by a gripper closed under 61 mm with its TCP within 6 mm of the
# bulb's axis and 10 mm of its grip height, turning with the gripper's heading; (3) the cap's end in the bore: with the
# holder held, it stops at the plungers' stop (5 mm); with the holder free, the holder rides up on it instead (the
# plungers' 0.8 N preload is more than the 0.59 N hanging below the slack slide) and turns with it; (4) the J rule
# (scoring.Bayonet) drives the catches; in the hooks the bulb cannot come lower than the hook floor; (5) let go, the
# bulb stays in the hooks, centred by the hook floors and their lips, else falls: onto its ring if over it, else to
# the table. Rigid arms, no swing.
HOLD_HOLDER = 2 * L.HOLDER_R + 0.004
HOLD_BULB = 2 * L.GRIP_R + 0.004
DT = 1.0 / L.CONTROL_HZ


def heading_of(tcp, tips) -> float:
    """The approach axis's heading (rad) in the top view."""
    a = np.asarray(tips, dtype=float) - np.asarray(tcp, dtype=float)
    return math.atan2(float(a[1]), float(a[0]))


class OfflineModel:
    def __init__(self) -> None:
        self.rule = S.Bayonet()
        self.reset()

    def reset(self) -> None:
        self.bulb = np.array([L.RING_X, L.RING_Y, L.TOP + L.BULB_START_Z])
        self.yaw = math.radians(L.BULB_START_YAW_DEG)
        self.rise = 0.0  # the holder pushed up the flex (the slack slide)
        self.twist = 0.0
        self.held = False
        self.grip = None  # (bulb minus TCP, bulb yaw minus hand heading) while carried
        self.rule.reset()

    def mouth(self) -> np.ndarray:
        return np.array([L.PEND_X, L.PEND_Y, L.TOP + L.MOUTH_Z + self.rise])

    def _holds_holder(self, tcp, opening) -> bool:
        tcp = np.asarray(tcp, dtype=float)
        h = float(tcp[2]) - float(self.mouth()[2])
        return bool(opening * rigstate.FULL_GAP < HOLD_HOLDER
                    and math.hypot(tcp[0] - L.PEND_X, tcp[1] - L.PEND_Y) <= 0.008 and 0.010 <= h <= L.HOLDER_LEN)

    def _holds_bulb(self, tcp, opening) -> bool:
        tcp = np.asarray(tcp, dtype=float)
        if opening * rigstate.FULL_GAP >= HOLD_BULB:
            return False
        if self.grip is not None:
            return True
        return bool(math.hypot(tcp[0] - self.bulb[0], tcp[1] - self.bulb[1]) <= 0.006
                    and abs(float(tcp[2]) - (self.bulb[2] - L.GRIP_FROM_END)) <= 0.010)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        left = right = None
        if read is not None:
            left, right = read[rigstate.ARMS.index("left")], read[rigstate.ARMS.index("right")]
        self.held = bool(left is not None and self._holds_holder(left[0], left[2]))
        yaw_before = self.yaw
        if right is not None and self._holds_bulb(right[0], right[2]):
            tcp = np.asarray(right[0], dtype=float)
            psi = heading_of(right[0], right[1])
            if self.grip is None:
                self.grip = (self.bulb - tcp, self.yaw - psi)
            self.bulb = tcp + self.grip[0]
            self.yaw = psi + self.grip[1]
        else:
            self.grip = None
        mouth = self.mouth()
        lateral = math.hypot(self.bulb[0] - mouth[0], self.bulb[1] - mouth[1])
        if self.grip is not None:
            if lateral <= 0.003:
                if self.held:
                    self.bulb[2] = min(self.bulb[2], mouth[2] + L.H_PL + L.C_MAX)
                else:
                    rest = float(self.bulb[2]) - (L.TOP + L.MOUTH_Z)
                    self.rise = min(L.SLACK_UP, max(0.0, rest - L.H_PL))
                    self.bulb[2] = min(self.bulb[2], L.TOP + L.MOUTH_Z + self.rise + L.H_PL + L.C_MAX)
                if not self.held and float(self.bulb[2] - self.mouth()[2]) >= L.H_PL - 1e-9:
                    self.twist += self.yaw - yaw_before  # the free holder turns with the bulb
                if self.rule.hooked:
                    self.bulb[2] = max(self.bulb[2], self.mouth()[2] + L.H_LOCK_END)
            elif not self.held:
                self.rise = 0.0
        else:
            if not self.held:
                self.rise = 0.0
                self.twist = 0.0
            if self.rule.hooked:
                self.bulb[:2] = self.mouth()[:2]  # the hook floors (from x) and their lips (from y) centre it
                self.bulb[2] = self.mouth()[2] + L.H_LOCK_END
            else:
                over_ring = math.hypot(self.bulb[0] - L.RING_X, self.bulb[1] - L.RING_Y) <= L.RING_OUT
                floor = L.BULB_START_Z if over_ring else L.BULB_LEN
                self.bulb[2] = min(self.bulb[2], L.TOP + floor)
        r = S.read(self.states())
        if r is not None:
            self.rule.update(r.lateral, r.h_end, r.tilt_deg, r.turn_deg)

    def _c(self) -> float:
        mouth = self.mouth()
        if math.hypot(self.bulb[0] - mouth[0], self.bulb[1] - mouth[1]) > 0.003:
            return 0.0
        return float(min(max(self.bulb[2] - mouth[2] - L.H_PL, 0.0), L.PL_TRAVEL))

    def states(self) -> dict:
        c = self._c()
        catch = L.CATCH_TRAVEL if self.rule.hooked else 0.0
        joints = {"slack": self.rise, "holder": self.twist, "plunger_a": c, "plunger_b": c, "catch_a": catch,
                  "catch_b": catch}
        return {PENDANT: mk.state_from_poses(PENDANT_PART, (L.PEND_X, L.PEND_Y, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                             joints=joints, source="offline"),
                BULB: mk.state_from_poses(BULB_PART, (*self.bulb, *mk.yaw_quat(self.yaw)), source="offline"),
                STAND: STAND_PART.rest_state(), RING: RING_PART.rest_state()}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(BULB), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def _reset_all() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    RULE.reset()


@rigstate.on_reset
def _new_episode() -> None:
    _reset_all()


@mk.on_reset
def _kit_reset() -> None:
    _reset_all()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the parts' poses and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The bulb locked in the hooks, never dropped, the holder not strained or wound, both hands off and the holder
    hanging still for 1 s; held 1 s. This runs first every control step, so it also feeds the tracker (and, without
    a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _reset_all()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    for n, spec in FIXED.items():
        if parts.get(n) is None:
            parts[n] = spec.rest_state()
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, "rule": (_OFFLINE.rule if offline else RULE).status(),
           **result.details}
    try:
        row["mouth"] = parts[PENDANT].link_pose("holder")[1]
        row["pendant_joints"] = {k: round(float(v), 5) for k, v in parts[PENDANT].joints.items()}
        row["bulb_origin"] = parts[BULB].origin
    except (AttributeError, KeyError, TypeError, ValueError):
        row["mouth"] = row["bulb_origin"] = None
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(BULB), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the bulb was dropped (the table or a stand touched after it left its ring) or is not locked in the
    hooks; 0.25 locked but the holder was pulled more than 15 mm down; 0.5 locked but the holder was turned more than
    90 degrees, or not both of hands off and hanging still; 1.0 locked, hands off, hanging still where it hung. The
    latest scored step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points, heights above
# the table; the pad tips are 13 mm past the TCP along the approach axis. Both grips are pitched side grips whose
# closing axis stays level: the left reaches forward and to the right, 40 degrees down, and pinches the holder 40 mm
# over its mouth; the right reaches forward and to the left, 20 degrees down, and pinches the bulb's globe 84 mm
# under the cap's end (10 mm under the globe's middle). The right turns the bulb by turning its heading from 75 to 90
# degrees about the bulb's axis (the TCP on the axis). The two grips stay 84 mm or more apart in height.
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
CLOSED = 0.0
HOLDER_GRIP = -(2 * L.HOLDER_R - 0.008) / 2  # 30 mm commanded across the 38 mm holder
BULB_GRIP = -(2 * L.GRIP_R - 0.0054) / 2  # 52 mm commanded across the globe's 57.4 mm


def grip_rotation(heading_deg: float, pitch_deg: float) -> np.ndarray:
    """A side grip: the approach axis (TCP z) heading ``heading_deg`` from +x toward +y in the top view and pitched
    ``pitch_deg`` down, the closing axis (TCP x) level. Pitched 90 degrees it is the stock straight-down grip turned
    by the heading."""
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    a = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    c = np.array([math.sin(h), -math.cos(h), 0.0])
    c = c - a * float(c @ a)
    c /= np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a])


def approach(heading_deg: float, pitch_deg: float) -> np.ndarray:
    return grip_rotation(heading_deg, pitch_deg)[:, 2]


def side_grip(heading_deg: float, pitch_deg: float) -> float:
    return tiltgrip.TiltedYaw(math.radians(heading_deg), grip_rotation(heading_deg, pitch_deg))


HEAD_L, PITCH_L = -60.0, 40.0
HEAD_R0, HEAD_R1, PITCH_R = 75.0, 75.0 + L.HOOK_DEG, 20.0
YAW_L = side_grip(HEAD_L, PITCH_L)
YAW_R0 = side_grip(HEAD_R0, PITCH_R)
YAW_R1 = side_grip(HEAD_R1, PITCH_R)
A_L = approach(HEAD_L, PITCH_L)

HOLD_L = np.array(L.holder_grip_point())
PRE_L = HOLD_L - 0.070 * A_L
BACK_L = HOLD_L - 0.100 * A_L
PARK_L = HOLD_L - 0.180 * A_L  # back along the approach, out of reach of the holder
LEFT_PULL = 0.007  # the left draws the holder down this much (commanded) while the bulb goes in
PULL_L = HOLD_L - np.array([0.0, 0.0, LEFT_PULL])  # under the grip point: the arm's give leaves a pull of about 1 N

STAND_GRIP = np.array([L.RING_X, L.RING_Y, L.BULB_START_Z - L.GRIP_FROM_END])
LIFT = 0.015  # out of the ring: the bulb's lowest point 11 mm over the ring's top
UNDER_Z = STAND_GRIP[2] + LIFT  # under the holder: the cap's end 13 mm under the mouth
CONTACT_Z = L.MOUTH_Z + L.H_PL - L.GRIP_FROM_END  # the cap's end on the plungers' tips
ENTER_Z = CONTACT_Z - 0.007
# The arms are soft springs at the TCP (GPU runs 1 and 2: about 150 to 250 N/m; the payload is not gravity
# compensated, so the 40 g bulb alone hangs about 2 mm under the command). The right is commanded 13 mm past the
# plungers' stop and the left draws the holder 7 mm (commanded) down against the flex while the bulb goes in, so the
# holder does not ride up under the push. A quasi-static model of both arms (tests: ArmGive) puts the cap's end at the
# plungers' 5 mm stop for arms of 120 to 300 N/m, with the holder 1.2 mm down to 5.1 mm up.
PUSH_OVER = 0.013
PUSH_Z = CONTACT_Z + L.C_MAX + PUSH_OVER
EASE_Z = CONTACT_Z + L.C_LOCK - 0.001  # 1 mm under the hook floor: the pins settle on it, the hand takes no weight
# Aim: in GPU run 1 (2026-11-18) the gripped holder's mouth sat at x -0.9, y -1.3 mm off its axis and the carried
# bulb's cap at x -0.4, y +0.3 mm, so the cap arrived 0.4 mm (x) and 1.5 mm (y) off the mouth. The right aims that
# much the other way so the cap goes up the middle of the bore.
AIM_SHIFT = np.array([-0.0004, -0.0015])
UNDER = np.array([L.PEND_X, L.PEND_Y]) + AIM_SHIFT
PARK_R_SHIFT = np.array([-0.050, 0.000, 0.040])  # aside: further back, toward the arms and up


def _w(d, p, finger, label, yaw):
    x, y, z = (float(v) for v in p)
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _at(z):
    return np.array([UNDER[0], UNDER[1], z])


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def right_moves(turn: bool = True, early: bool = False):
    """The right arm: behind the bulb, fingers either side of the globe, grip, lift out of the ring, under the
    holder, up into the mouth, push to the plungers' stop, turn 15 degrees, ease down into the hooks, let go, back
    out, aside."""
    a0 = approach(HEAD_R0, PITCH_R)
    g = STAND_GRIP
    plan = [
        _w(2.0, g - 0.070 * a0, OPEN, "behind the bulb on its stand", YAW_R0),
        _w(1.2, g, OPEN, "fingers either side of the globe", YAW_R0),
        _w(0.3, g, OPEN, "settle", YAW_R0),
        _w(1.5, g, BULB_GRIP, "grip the globe", YAW_R0),
        _w(1.2, g + (0.0, 0.0, LIFT), BULB_GRIP, "lift it out of the ring", YAW_R0),
        _w(2.0, _at(UNDER_Z), BULB_GRIP, "under the holder, the pins lined up with the J's legs", YAW_R0),
        _w(1.5, _at(ENTER_Z), BULB_GRIP, "up into the mouth", YAW_R0),
        _w(1.0, _at(CONTACT_Z), BULB_GRIP, "the cap's end on the plungers", YAW_R0),
        _w(1.0, _at(PUSH_Z), BULB_GRIP, "push up to the plungers' stop", YAW_R0),
        _w(0.5, _at(PUSH_Z), BULB_GRIP, "hold it up", YAW_R0),
    ]
    yaw, head = YAW_R0, HEAD_R0
    if early:
        plan.append(_w(1.5, _at(PUSH_Z), OPEN, "let go before turning", yaw))
        z = PUSH_Z
    else:
        if turn:
            yaw, head = YAW_R1, HEAD_R1
            plan += [_w(1.5, _at(PUSH_Z), BULB_GRIP, "turn it 15 degrees at the top", yaw),
                     _w(0.5, _at(PUSH_Z), BULB_GRIP, "hold", yaw)]
        plan += [
            _w(1.0, _at(EASE_Z), BULB_GRIP, "ease down: the pins drop into the hooks", yaw),
            _w(0.5, _at(EASE_Z), BULB_GRIP, "settle", yaw),
            _w(1.5, _at(EASE_Z), OPEN, "let go of the bulb", yaw),
        ]
        z = EASE_Z
    a = approach(head, PITCH_R)
    plan += [
        _w(1.2, _at(z) - 0.100 * a, OPEN, "back out", yaw),
        _w(1.5, _at(z) - 0.140 * a + PARK_R_SHIFT, OPEN, "aside", yaw),
    ]
    return plan


def left_moves(right):
    """The left arm: beside the holder, fingers either side of it, grip it 40 mm over the mouth, draw it down a
    little against the flex and hold it steady until the right hand has let go of the locked bulb, ease the pull off,
    let go, back out, aside."""
    plan = [
        _w(2.0, PRE_L, OPEN, "beside the holder", YAW_L),
        _w(1.2, HOLD_L, OPEN, "fingers either side of the holder", YAW_L),
        _w(0.3, HOLD_L, OPEN, "settle", YAW_L),
        _w(1.5, HOLD_L, HOLDER_GRIP, "grip the holder", YAW_L),
        _w(0.5, PULL_L, HOLDER_GRIP, "draw it down a little against the flex", YAW_L),
    ]
    release_at = _until(right, "let go") + 0.3
    plan.append(_w(release_at - _duration(plan), PULL_L, HOLDER_GRIP, "hold it steady while the bulb goes in",
                   YAW_L))
    plan += [
        _w(0.5, HOLD_L, HOLDER_GRIP, "ease the pull off", YAW_L),
        _w(1.2, HOLD_L, OPEN, "let go of the holder", YAW_L),
        _w(1.2, BACK_L, OPEN, "back out", YAW_L),
        _w(1.5, PARK_L, OPEN, "aside", YAW_L),
    ]
    return plan


# Wrong ways (counterfactuals, run on the GPU for the README; the offline model predicts them roughly):
#   noturn  pushed up and eased down without the turn: the pins come back down the legs and the bulb falls
#   early   let go at the top before the turn: the plungers push the bulb out and it falls
VARIANTS = ("noturn", "early")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = right_moves(turn=variant != "noturn", early=variant == "early")
    return {"left": left_moves(right), "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      noturn  pushed up and eased down without the turn: the bulb falls out
      early   let go at the top before the turn: the bulb falls out"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
