"""On the station table, real size: an OHAUS 750-S0 class triple beam balance (610 g, 0.1 g; a base with a pillar and
knife edge, a beam that turns on it with a self-righting spring and a damper, a 15.2 cm pan on a stem from the beam's
left end, three parallel beams to the right with their weights, a pointer against a fixed scale post) and, on the table
in front of it, a granite specimen of 236.4 g.

Goal: the specimen on the pan, the 100 g rider in the 200 g notch of the 0 to 500 g beam, the 10 g rider in the 30 g notch
of the 0 to 100 g beam, the poise slid along the smooth 0 to 10 g beam until the pointer rests on the zero mark (6.4 g),
the hands clear, everything at rest for 1 s. The weights are pushed along their beams with closed fingers (the
notched riders over a detent, the poise on a little friction); the beam swings and damps over a few seconds, so the poise
is crept in.

The balance (beam on a revolute joint, three weights on prismatic joints, fixed base) and the specimen are mechkit
assemblies (mechkit.py, next to this file). The arms are read through rigstate.py (the let-go checks).

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
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s windows

INSTRUCTION = (
    "Weigh the rock on the triple beam balance: set it on the pan, move the 100 g and then the 10 g rider along their "
    "beams, slide the poise until the pointer rests on the zero mark, and take your hands off."
)

CW_D = P.tune()  # the tare weight's place, so the empty beam hangs at zero
BALANCE = P.build(CW_D)
BEAM_SPRING = P.beam_spring(CW_D)  # N m/rad: K_BEAM net with the rock on the pan, plus the parts' gravity term
ROCK = P.specimen()
ROCK_CM = ROCK.center_of_mass("base")  # in the rock's frame (origin on the table under its centre)
OBJECTS = [BALANCE, ROCK]


def detent_pull_target(q: float, notches, capture: float, k: float) -> float:
    """The drive target of a capped detent: within ``capture`` of a notch the target is moved toward it by at most
    RIDER_PULL / k, so the stiff spring pulls with at most RIDER_PULL (about 2 N) and holds the rider IN the notch once
    it is within RIDER_PULL / k of it; between notches the target is where the rider is (it stays where it was left)."""
    q = float(q)
    near = min((float(n) for n in notches), key=lambda n: abs(n - q))
    if abs(near - q) > capture:
        return q
    cap = L.RIDER_PULL / k
    return q + max(-cap, min(cap, near - q))


def roberval_target(bal, rock) -> float:
    """The rule that stands for the pan's Roberval linkage: a load lying on the plate turns the beam as if it sat on
    the pan's axis, wherever on the plate it lies. Without it 1 mm of placement along the beam is 2 g at the pan
    (GPU run 1: the rock lay 7 mm toward the knife edge, 14 g light). The beam spring's target is offset by the
    torque of the rock's offset from the axis (its weight times its centre of mass's distance from the axis along
    the beam, in the beam's frame) over the spring rate. 0 unless the rock rests on the plate."""
    try:
        r, p = bal.link_pose(S.BEAM)
        off = bal.spec.body_offset(S.BEAM)
        cm = rock.to_world(ROCK_CM)
        base = rock.to_world(np.zeros(3))
        loc_cm = (cm - p) @ r + off
        loc_base = (base - p) @ r + off
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0.0
    if not (np.all(np.isfinite(loc_cm)) and np.all(np.isfinite(loc_base))):
        return 0.0
    if math.hypot(float(loc_cm[0]), float(loc_cm[1]) - L.LA) > L.ON_PAN_R:
        return 0.0
    if abs(float(loc_base[2]) - L.PAN_Z) > 0.002:  # not lying on the plate (held over it, or elsewhere)
        return 0.0
    th = float(bal.joints.get(S.BEAM, 0.0))
    torque = -L.SPEC_MASS * L.G * (L.LA - float(loc_cm[1])) * math.cos(th)  # what a load on the axis adds
    lim = math.radians(L.STOP_DEG) - 0.005
    return float(np.clip(torque / BEAM_SPRING, -lim, lim))


@mk.on_step
def _detents(states) -> None:
    """The notched riders' detents (rule-based, as meter-test-switch's, with a capped pull): a rider within its
    capture distance of a notch is drawn into it (about 2 N to pull it out), between notches it stays where it was
    left. And the Roberval rule (roberval_target). Set every step because mechkit's targets go back to 0 at reset.
    The poise has no detent: friction holds it."""
    st = states.get(S.BALANCE)
    if st is None:
        return
    for name, step, n in ((S.R100, L.R100_STEP, L.R100_N), (S.R10, L.R10_STEP, L.R10_N)):
        q = st.joints.get(name)
        if q is None:
            continue
        mk.set_targets(BALANCE, name, detent_pull_target(q, [k * step for k in range(n + 1)], L.RIDER_CAPTURE[name],
                                                         L.RIDER_DETENT_K[name]))
    rock = states.get(S.ROCK)
    mk.set_targets(BALANCE, S.BEAM, roberval_target(st, rock) if rock is not None else 0.0)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and knows no joints. Offline only (never in the simulator), this
# model gives the scorer: THE SPECIMEN gripped (fingers closed on it, TCP over its centre) moves with the hand; let go
# over the pan it settles on the pan (and then turns with the beam), elsewhere it drops to the table. THE WEIGHTS: a
# closed hand's pad block beside a weight pushes it along its beam (the others stay); a rider let go within its
# capture distance of a notch is drawn onto it. THE BEAM: quasi-static torque balance of the specimen on the pan against
# the three weights, clamped to the stops, followed with the damper's time constant.
# From GPU run 2's trace (2026-11-21, job 20261121T110606-00), the push itself: the pad meets a weight's left face at
# its top edge with the beam pan side down (the face leans toward the pad) and at the pad's tip with it pan side up,
# so where a weight ends for a given hand depends on the beam angle (10 mm between the stops); the pad pushes from
# PAD_BEND short of its drawn face (wrist bend under the push; run 2's +8 samples); and the push turns the beam pan
# side up by its force times the contact's height over the axis line (PUSH_F: out of a notch's capture, between
# notches, the poise), which is what took the beam to +8 in the 10 g rider's push and held it there.
PAD_BEND = 0.003
PUSH_F = {"detent": 2.2, "between": 1.5, "poise": 0.3}  # N
PAD_HALF_Y = 0.008
PAD_HALF_X = 0.010
PAD_LEN = 0.038
TAU_BEAM = L.C_BEAM / L.K_BEAM
CLOSED_OPENING = 0.15
GRIP_OPENING = (0.15, 0.60)


def _axis_z(d: float, theta: float) -> float:
    """Height over the table top of the beam's axis line at ``d`` right of the knife edge for beam angle ``theta``."""
    return L.HZ - d * math.sin(theta)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.z = L.SPEC_X, L.SPEC_Y, 0.0  # the specimen's frame origin (z over the table top)
        self.held: tuple | None = None
        self.pan_rel: tuple | None = None  # (dx, dy over the pan's axis) while it rests on the pan
        self.theta = 0.0
        self.q = {S.R100: 0.0, S.R10: 0.0, S.POISE: 0.0}
        self.push_torque = 0.0

    # -- the specimen --
    def _grips(self, tcp, tips, opening) -> bool:
        if not (GRIP_OPENING[0] <= opening <= GRIP_OPENING[1]):
            return False
        return bool(abs(float(tcp[0]) - self.x) <= 0.020 and abs(float(tcp[1]) - self.y) <= 0.015
                    and 0.008 <= float(tcp[2]) - L.TOP - self.z <= 0.045)

    def _pan_centre(self):
        """(y, z over the table) of the pan's axis at plate level for the current beam angle, and the cos/sin."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        up = L.PAN_Z - L.HZ
        return L.YP + L.LA * c - up * s, L.HZ + L.LA * s + up * c

    def _rock_step(self, read) -> None:
        holding = [r for r in read if self._grips(*r)]
        if holding:
            tcp = holding[0][0]
            if self.held is None:
                self.held = (self.x - float(tcp[0]), self.y - float(tcp[1]), self.z - (float(tcp[2]) - L.TOP))
            self.pan_rel = None
            self.x, self.y = float(tcp[0]) + self.held[0], float(tcp[1]) + self.held[1]
            self.z = float(tcp[2]) - L.TOP + self.held[2]
            return
        if self.held is not None:  # let go: it drops onto what is under it
            self.held = None
            py, _ = self._pan_centre()
            if math.hypot(self.x - L.X0, self.y - py) <= L.ON_PAN_R and self.z >= L.PAN_Z - 0.01:
                self.pan_rel = (self.x - L.X0, self.y - py)
            else:
                self.z = 0.0
        if self.pan_rel is not None:  # rides on the pan: the plate's top, turned with the beam
            c, s = math.cos(self.theta), math.sin(self.theta)
            up = L.PAN_Z - L.HZ
            ry = L.LA + self.pan_rel[1]
            self.x = L.X0 + self.pan_rel[0]
            self.y = L.YP + ry * c - up * s
            self.z = L.HZ + ry * s + up * c

    # -- the weights --
    def _push(self, name: str, beam: str, length: float, top: float, read) -> bool:
        pushed = False
        dx = L.BEAM_DX[beam]
        th = self.theta
        c, s = math.cos(th), math.sin(th)
        d = L.ZERO_D + self.q[name]
        zax = L.TOP + _axis_z(d, th)
        for tcp, tips, opening in read:
            if opening > CLOSED_OPENING:
                continue
            if abs(float(tcp[0]) - (L.X0 + dx)) > PAD_HALF_X + L.RIDER_W / 2:
                continue
            tip_z = float(tips[2])
            if not (zax + 0.0055 <= tip_z <= zax + top):
                continue
            py = float(tcp[1])
            # the contact's height over the axis line in the beam's frame: the weight's top edge with the beam pan
            # side down, the pad's tip with it pan side up
            if th < 0:
                h = top
            else:
                h = float(np.clip(-(py - L.YP) * s + (tip_z - L.TOP - L.HZ) * c, 0.0, top))
            face = L.YP - (d - length / 2) * c - h * s  # the weight's left face at that height
            lo, hi = py - PAD_HALF_Y + PAD_BEND, py + PAD_HALF_Y
            if hi > face - length * c and lo < face and py >= face - length * c / 2:  # overlaps: push it clear
                d_face = (L.YP - h * s - lo) / c
                q_new = float(np.clip(d_face + length / 2 - L.ZERO_D, 0.0, self._upper(name)))
                if q_new > self.q[name]:
                    if name == S.POISE:
                        f = PUSH_F["poise"]
                    else:
                        step = L.R100_STEP if name == S.R100 else L.R10_STEP
                        k = round(self.q[name] / step)
                        f = PUSH_F["detent" if abs(self.q[name] - k * step) <= L.RIDER_CAPTURE[name] else "between"]
                    self.push_torque += f * h
                    self.q[name] = q_new
                elif name != S.POISE:
                    # holding a rider past its notch against the detent's pull (capped at 2 N): the pad takes that
                    # pull at height h, which keeps the beam pan side up while the hand stays
                    step = L.R100_STEP if name == S.R100 else L.R10_STEP
                    past = self.q[name] - round(self.q[name] / step) * step
                    if 0.0 < past <= L.RIDER_CAPTURE[name]:
                        self.push_torque += min(L.RIDER_PULL, L.RIDER_DETENT_K[name] * past) * h
                pushed = True
        return pushed

    @staticmethod
    def _upper(name: str) -> float:
        return {S.R100: L.R100_STEP * L.R100_N, S.R10: L.R10_STEP * L.R10_N, S.POISE: L.POISE_SPAN}[name]

    def _weights_step(self, read) -> None:
        for name, beam, length, top in ((S.R100, "b500", L.RIDER_L, L.RIDER_TOP), (S.R10, "b100", L.RIDER_L, L.RIDER_TOP),
                                        (S.POISE, "poise", L.POISE_L, L.POISE_TOP)):
            pushed = self._push(name, beam, length, top, read)
            if not pushed and name != S.POISE:
                step = L.R100_STEP if name == S.R100 else L.R10_STEP
                k = round(self.q[name] / step)
                if abs(self.q[name] - k * step) <= L.RIDER_CAPTURE[name]:
                    self.q[name] = float(k * step)

    # -- the beam --
    def _beam_step(self) -> None:
        left = 0.0
        if self.pan_rel is not None:
            left = L.SPEC_MASS * L.LA  # the Roberval rule: a load on the plate acts at the pan's axis
        right = L.R100_MASS * self.q[S.R100] + L.R10_MASS * self.q[S.R10] + L.POISE_MASS * self.q[S.POISE]
        eq = (-(left - right) * L.G + self.push_torque) / L.K_BEAM  # pan side down is a negative angle
        stop = math.radians(L.STOP_DEG)
        eq = float(np.clip(eq, -stop, stop))
        self.theta += (eq - self.theta) / (TAU_BEAM * L.CONTROL_HZ)
        self.theta = float(np.clip(self.theta, -stop, stop))

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        self.push_torque = 0.0
        self._rock_step(read)
        self._weights_step(read)
        self._beam_step()
        if self.pan_rel is not None:
            self._rock_step([])

    def rock_state(self):
        c, s = math.cos(self.theta / 2), math.sin(self.theta / 2)
        tilt = (s, 0.0, 0.0, c) if self.pan_rel is not None else (0.0, 0.0, 0.0, 1.0)
        return mk.state_from_poses(ROCK, (self.x, self.y, L.TOP + self.z, *tilt), source="offline")

    def balance_state(self):
        joints = {S.BEAM: self.theta, **self.q}
        return mk.state_from_poses(BALANCE, (L.X0, L.YP, L.TOP, 0.0, 0.0, 0.0, 1.0), joints=joints, source="offline")


_OFFLINE = OfflineModel()


def no_simulator(rock) -> bool:
    return mk.env() is None and getattr(rock, "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, beam deg, reading g, poise mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: the balance's joints and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Specimen on the pan, both riders in their notches, reading within 0.5 g, pointer within 1 degree, beam at rest,
    the hands clear, no latched fall; held 1 s. Runs first every control step, so it also feeds the tracker (and,
    without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    bal = objects.get(S.BALANCE)
    rock = objects.get(S.ROCK)
    arms = rigstate.arms()
    if no_simulator(rock):
        _OFFLINE.step(arms)
        rock = _OFFLINE.rock_state()
        bal = _OFFLINE.balance_state()
    result = _TRACKER.update(bal, rock, arms)
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
    row["source"] = getattr(rock, "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["beam_deg"], d["reading_g"], d["poise_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the specimen is on the pan, let go and still (and never fell off the table); 0.25 so; 0.5 with both
    riders in their notches; 0.75 with the reading within 2 g, the pointer within 3 degrees and the beam at rest; 1.0
    with the reading within 0.5 g, the pointer within 1 degree and the hands clear of the instrument. The score of the
    latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. Finger targets: the joint
# value is minus half the commanded gap. The right arm does everything (the left arm parks); the weights are pushed with
# closed fingers (yaw 0: the fingers close along y, the closed pad block is 16 mm along y and 20 mm along x).
#
# What the GPU runs showed about pushing a weight along a beam that can turn (runs 1 and 2, 2026-11-21):
#  * On the -8 stop a weight's left face leans toward the pad, so the pad meets it at its top edge (34 mm over the
#    axis line), and the weight climbs the tilted beam under the pad, so friction pulls it down: pushing a rider out of
#    a notch (about 2 N) turns the beam pan side up by 0.07 to 0.15 N m. The 100 g push lifted the beam off the stop
#    against 110 to 135 g (run 2, steps 622-680). With the 100 g rider in notch 2 only 36 g (0.043 N m) is left on the
#    pan side, so the 10 g rider's push out of its zero notch takes the beam to +8 at once (run 2, steps 856-880).
#    Steadying the pan with the left hand does not fit: at every point of the plate the two arms' wrists overlap
#    (capsule proxy, 6 to 30 mm) while the right hand is at the start of a push.
#  * On +8 the face leans away and the pad meets it at the pad's tip: the push holds the beam on +8 all the way
#    (run 2, steps 880-944), and the rider's position follows the hand as q10 = -208.9 mm - TCP y (four samples,
#    +-0.3 mm; on -8 it is -199 mm - TCP y). Run 2's push ended where that put the rider at 36.6 mm with no load; it
#    stalled at 30.6 mm, 0.6 mm past notch 2, the hand 5.7 mm behind its target pushing 1.1 N (0.19 N/mm): too little
#    to pull the rider out of notch 2's detent (2 N).
# So the 10 g push is planned for the beam on +8 and runs on to a TCP that would put the rider at 46 mm (1 mm past
# notch 3) with no load: 15 mm past the stall point, so the lag can build the 2 N the detent needs; once out of notch
# 2 the rider runs on (downhill on +8) into notch 3's capture (40 to 50 mm) and is drawn to 45 mm. Then the pad draws
# back 15 mm at push height before rising: with the pad still down, the beam's return to -8 would carry the rider
# about 10 mm further (the -8 relation), out of notch 3's capture (offline model, end points 47 mm and over).
PAD = 0.013  # the pad tips are this far past the TCP
OPEN = -0.04695  # 94 mm
WIDE = -0.035  # 70 mm: open round the 48 mm specimen
GRIP = -0.019  # 38 mm commanded on the 48 mm specimen: about 5 mm a finger short of it
CLOSED = 0.0
SPEC_GRIP_Z = 0.010 + PAD  # TCP height with the pad tips 10 mm over the table
CARRY_Z = 0.190  # TCP while carrying: the specimen's underside 167 mm up, 22 mm over the pan
SET_Z = L.PAN_Z + 0.003 + 0.010 + PAD  # underside 3 mm over the plate
CLEAR_Z = 0.200
PUSH_SPEED = 0.020  # m/s along a beam (the poise's first 24 mm)
RIDER_SPEED = 0.010  # m/s for the riders: the detent's held target resists about k v dt / 2 (0.5 N at 2000 N/m)
CREEP_SPEED = 0.0045  # m/s for the poise's last 40 mm (0.45 g per second: the damper follows within 1 degree)
# The pad tip's height. Pushing a weight along -y at a height over the knife edge turns the beam pan side up; near
# balance the push alone (0.3 to 1 N at 20 to 30 mm) is worth several grams and takes the beam onto its +8 degree
# stop (GPU run 1: the beam flipped to +8 during the 10 g rider's push and the pad, aimed for -8, passed over the 10 g
# rider at 31 mm and over the poise at 48 mm). So the tip is aimed at the middle of the window that holds on EITHER
# stop: over the bar (5 mm over the axis) with the beam at -8 and under the weight's top (34 mm) with it at +8, at
# the weight's left face (``pad_tip_z``). For the 100 g rider, whose push ends 88 mm out where that window is only
# 5 mm, the beam stays on its -8 stop (36 g still on the pan side, far more than the push): its window is taken
# for -8 to +4 degrees.
SIN8 = math.sin(math.radians(L.STOP_DEG))
TIP_POISE_Z = L.HZ + (0.005 + L.POISE_TOP) / 2  # the poise's pad tip height over the table, constant: 0.1195
PARK = {"left": (0.20, 0.27, 0.16), "right": (0.20, -0.30, 0.20)}
# The 10 g rider's push on the +8 stop (see above): the TCP y that puts the rider at q10 is -0.2089 - q10 (run 2).
R10_P8_OFFSET = 0.2089
R10_PUSH_TO = 0.046  # m: the rider position the push's end TCP stands for, on +8 with no load
R10_BACK_OFF = 0.015  # m: drawn back before rising (more than the 10 mm the beam's return to -8 moves the rider's face)
R10_END_TIP = 0.0175  # m over the axis line: the pad tip at the push's end, 8 mm under the rider's top on +8
CLEARANCE_Y = 0.012  # start of a push: the pad's face this far from the weight
# The poise's creep ends 1.5 mm further than run 2's. Run 2 left it at 65.6 mm with the beam held at -5.3 degrees by
# the push (10 to 14 g short then): the pad met the poise's top edge, 3.1 mm nearer the pad than its foot. Near
# balance (as now, the 10 g rider in notch 3) the push holds the beam near level or a little over (the pad meets the
# poise lower and the face no longer leans toward it), 2 to 4 mm less poise for the same hand: 61.5 to 63.5 mm, so
# 63 to 65 mm with this. Within 59 to 69 mm the reading is within 0.5 g.
POISE_EXTRA = 0.0015
SHORT = {"r100": 0.002, "r10": 0.0015}  # the pad stops this far short of the notch: the detent pulls the rider home


def pad_tip_z(d_face: float, top: float, theta_hi: float) -> float:
    """Pad tip height over the table (constant beam-axis reference HZ) in the middle of the window that keeps it on
    a weight's left face ``d_face`` right of the knife edge for any beam angle from -8 degrees to ``theta_hi``:
    above the bar's top at -8, under the weight's top at ``theta_hi``."""
    lo = 0.005 + d_face * SIN8
    hi = top - d_face * math.sin(theta_hi)
    return L.HZ + (lo + hi) / 2


def _w(d, p, finger, label, yaw=0.0):
    return Waypoint(d, (float(p[0]), float(p[1]), float(L.TOP + p[2])), finger, label, yaw)


def _tcp_y(face_y: float) -> float:
    """TCP y of a closed hand whose -y pad face is at ``face_y`` (pushing right, toward -y)."""
    return face_y + PAD_HALF_Y


def rider_moves(beam: str, k: int, step: float, length: float, label: str, theta_hi: float, settle: float = 0.5,
                end_y: float | None = None, end_tip: float | None = None, back_off: float = 0.0):
    """Push a notched rider from its zero notch toward notch ``k``: the closed pad block comes down on its left, 12 mm
    clear, pushes it right (-y) to within SHORT of the notch (inside the capture distance: the detent pulls it the
    rest of the way), and rises away from it. Pad heights from ``pad_tip_z`` (beam anywhere from -8 degrees to
    ``theta_hi``). ``back_off``: draw the pad back (+y) this far at push height before rising, so the beam turning back
    pan side down cannot carry the rider further under a pad that is still down (10 mm of q between the stops)."""
    name = "r100" if beam == "b500" else "r10"
    x = L.X0 + L.BEAM_DX[beam]
    face0 = L.YP - L.ZERO_D + length / 2  # the rider's left face at its zero notch
    d1 = L.ZERO_D + k * step - SHORT[name]
    face1 = L.YP - d1 + length / 2
    z0 = pad_tip_z(L.ZERO_D - length / 2, L.RIDER_TOP, theta_hi) + PAD
    z1 = pad_tip_z(d1 - length / 2, L.RIDER_TOP, theta_hi) + PAD
    y0 = _tcp_y(face0 + CLEARANCE_Y)
    y1 = _tcp_y(face1)
    if end_y is not None:
        y1 = end_y
    if end_tip is not None:
        z1 = L.HZ + end_tip + PAD
    return [
        _w(2.5, (x, y0, CLEAR_Z), CLOSED, f"over the {label}, left of it, fingers closed"),
        _w(1.5, (x, y0, z0), CLOSED, f"down beside the {label}'s left face"),
        _w(0.5, (x, y0, z0), CLOSED, "settle"),
        _w(abs(y0 - y1) / RIDER_SPEED, (x, y1, z1), CLOSED, f"push the {label} to its notch"),
        _w(settle, (x, y1, z1), CLOSED, "settle: the detent pulls it home"),
    ] + ([_w(0.6, (x, y1 + back_off, z1), CLOSED, "draw back off it")] if back_off > 0 else []) + [
        _w(1.0, (x, y1 + max(back_off, 0.020), CLEAR_Z), CLOSED, "straight up, away from it"),
    ]


def poise_moves(q_end: float, creep_from: float = 0.024):
    """Push the poise right along the smooth beam: quickly to ``creep_from`` (the beam is still on its stop), then
    creep to ``q_end`` while the beam comes off the stop and settles, then dwell."""
    x = L.X0 + L.BEAM_DX["poise"]
    face0 = L.YP - L.ZERO_D + L.POISE_L / 2
    y0 = _tcp_y(face0 + CLEARANCE_Y)
    ya = _tcp_y(face0 - creep_from)
    yb = _tcp_y(face0 - q_end) - 0.0005 - POISE_EXTRA
    z = TIP_POISE_Z + PAD  # pad_tip_z(face, POISE_TOP, +8 degrees) at every face position
    return [
        _w(2.5, (x, y0, CLEAR_Z), CLOSED, "over the poise, left of it, fingers closed"),
        _w(1.5, (x, y0, z), CLOSED, "down beside the poise's left face"),
        _w(0.5, (x, y0, z), CLOSED, "settle"),
        _w(abs(y0 - ya) / PUSH_SPEED, (x, ya, z), CLOSED, "push the poise 24 mm while the beam is still on its stop"),
        _w(abs(ya - yb) / CREEP_SPEED, (x, yb, z), CLOSED, "creep it to 64 mm while the pointer comes off the stop"),
        _w(3.0, (x, yb, z), CLOSED, "wait"),
        _w(1.5, (x, yb + 0.030, CLEAR_Z), CLOSED, "straight up, clear: the beam comes off the push and settles on zero"),
    ]


def specimen_moves():
    px, py = L.SPEC_X, L.SPEC_Y
    qx, qy = L.X0, L.YP + L.LA
    return [
        _w(3.0, (px, py, 0.100), WIDE, "over the specimen, fingers open"),
        _w(2.0, (px, py, SPEC_GRIP_Z), WIDE, "down beside it"),
        _w(1.5, (px, py, SPEC_GRIP_Z), GRIP, "grip it across its 48 mm"),
        _w(0.5, (px, py, SPEC_GRIP_Z), GRIP, "settle the grip"),
        _w(2.0, (px, py, CARRY_Z), GRIP, "lift it"),
        _w(3.0, (qx, qy, CARRY_Z), GRIP, "carry it over the pan"),
        _w(1.5, (qx, qy, SET_Z), GRIP, "lower it to 3 mm over the plate"),
        _w(1.5, (qx, qy, SET_Z), WIDE, "let go"),
        _w(1.5, (qx, qy, CLEAR_Z), WIDE, "straight up"),
        _w(5.0, (qx, qy, CLEAR_Z), WIDE, "wait: the beam drops onto its stop"),
    ]


def park_moves(side: str, duration: float):
    return [_w(2.5, PARK[side], OPEN, "away"), _w(max(0.1, duration - 2.5), PARK[side], OPEN, "hold")]


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


VARIANTS = ("no_poise", "no_riders", "wrong_notch")


def plans(variant: str = "") -> dict:
    """Both arms' plans. The right arm does all the work; the left arm only parks.

    no_poise: the riders set, the poise left at zero (reading 230 g: 0.5).
    no_riders: the specimen on the pan and nothing else (0.25).
    wrong_notch: the 100 g rider put in the 300 g notch and the 10 g rider left at zero: the beam goes to the other stop (0.25)."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = specimen_moves()
    away = [_w(2.5, PARK["right"], CLOSED, "away, fingers closed"), _w(8.0, PARK["right"], CLOSED, "hold")]
    if variant == "no_riders":
        right += away
    elif variant == "wrong_notch":
        right += rider_moves("b500", 3, L.R100_STEP, L.RIDER_L, "100 g rider", math.radians(4.0)) + away
    else:
        right += rider_moves("b500", L.K100_ANS, L.R100_STEP, L.RIDER_L, "100 g rider", math.radians(4.0))
        right += rider_moves("b100", L.K10_ANS, L.R10_STEP, L.RIDER_L, "10 g rider", math.radians(L.STOP_DEG),
                             settle=0.3, end_y=-(R10_P8_OFFSET + R10_PUSH_TO), end_tip=R10_END_TIP,
                             back_off=R10_BACK_OFF)
        if variant == "no_poise":
            right += away
        else:
            right += poise_moves(L.POISE_ANS) + away
    return {"left": park_moves("left", _duration(right)), "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_poise     riders set, poise left at zero: 0.5
      no_riders    specimen on the pan, nothing else: 0.25
      wrong_notch  100 g rider in the 300 g notch: 0.25"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
