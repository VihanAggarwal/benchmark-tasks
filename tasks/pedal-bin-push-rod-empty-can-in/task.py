"""On the station table, real size: a Brabantia NewIcon 3 L pedal bin (264 mm high, 168 mm across, 0.83 kg) stands
loose at x 330 mm, y 0, its pedal toward the arms and its lid hinged at the back. An empty 330 ml drink can (66 mm
across, 115 mm tall, 14 g) stands upright on the table at x 260 mm, y 240 mm, left of the bin.

Goal: the can in the bin and the lid shut again, the lid opened only with the pedal. The pedal lifts the lid only while
it is held down, so one arm holds the pedal down while the other drops the can in, and the pedal is let go last.

The bin is one loose mechkit articulation (mechkit.py, next to this file): the body (a wall of 16 slabs, the inner
bucket's floor, a slot in the back wall), the pedal lever on a hinge under the floor and the lid on its hinge. The push
rod is a second loose part standing in the slot. The coupling is contact only: the lever's rear nose lifts the rod, the
rod's flat top lifts a pin under the lid's hinge; let go, the lid's weight pushes them back and the lid hinge's damping
is the soft close. No spring and no software rule. The arms are read through rigstate.py (the lid-touch check).
tiltgrip.py wraps ``yam_arena.demo.grip_rotation`` so that the demo's waypoints can carry pitched grips.

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
from scoring import BIN, CAN, ROD  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of shut-and-still

INSTRUCTION = "Throw the empty can into the pedal bin, opening the lid with the pedal only."


# -- the parts --------------------------------------------------------------------------------------------------------
PEDAL = mk.hinge("pedal", "-Y", (L.PIVOT_X, 0.0, L.PIVOT_Z), P.lever_shapes(), upper=L.LEVER_UPPER,
                 mass=L.LEVER_MASS, damping=L.LEVER_DAMP, color=L.PEDAL_GREY)
LID = mk.hinge("lid", "Y", (L.HINGE_X, 0.0, L.HINGE_Z), P.lid_shapes(), upper=L.LID_UPPER, mass=L.LID_MASS,
               damping=L.LID_DAMP, color=L.LID_GREY)
BIN_PART = mk.assembly(BIN, origin=(L.BIN_X, L.BIN_Y, L.TOP), mass=L.BODY_MASS, shapes=P.body_shapes(),
                       joints=[PEDAL, LID], color=L.GREY, friction=L.BIN_FRICTION, tight_fit=True)
ROD_PART = mk.assembly(ROD, origin=(L.BIN_X + L.ROD_X, L.BIN_Y, L.TOP + L.ROD_BOT), mass=L.ROD_MASS,
                       shapes=P.rod_shapes(), color=L.ROD_COLOR, friction=L.ROD_FRICTION, tight_fit=True)
CAN_PART = mk.assembly(CAN, origin=(L.CAN_X, L.CAN_Y, L.TOP + L.CAN_H / 2), mass=L.CAN_MASS,
                       shapes=P.can_shapes(), color=L.CAN_COLOR, friction=L.CAN_FRICTION)
OBJECTS = [BIN_PART, ROD_PART, CAN_PART]
SPECS = {BIN: BIN_PART, ROD: ROD_PART, CAN: CAN_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows no pedal, rod or lid. Offline only (never in the simulator), this model gives the scorer:
# (1) the pedal pressed by a closed gripper whose lowest finger corner is over the plate, to that corner's depth (the
# lever's stop at most); (2) the rod and the lid where that lever angle puts them (layout.rod_from_lever, lid_from_rod);
# let go, the lid closes at the rate its weight and the hinge damping give, the rod and the lever following it; (3) the
# can carried by a gripper closed round it and, let go, falling under gravity: to the bucket floor if it is over the
# opening with the lid 30 degrees or more open, onto the lid if it is over the bin otherwise, to the table elsewhere.
HOLD_GAP = 2 * L.CAN_R + 0.004  # a gripper holds the can while its pads are under 70 mm apart (60 commanded)
DT = 1.0 / L.CONTROL_HZ


def corner_z(tcp, tips) -> float:
    """The lowest point of a finger's end: the pad-tip point less half the pad's width times the approach axis's
    lean (a pitched grip meets a plate with its rear corner first)."""
    a = np.asarray(tips, dtype=float) - np.asarray(tcp, dtype=float)
    n = float(np.linalg.norm(a)) or 1.0
    return float(tips[2]) - rigstate.FINGER_HALF_WIDTH * float(np.hypot(a[0], a[1])) / n


def closing_rate(lid_rad: float) -> float:
    """rad/s: the lid's weight against the hinge damping (the soft close)."""
    torque = L.LID_MASS * L.G * (-L.LID_CG_X * math.cos(lid_rad) - L.LID_CG_Z * math.sin(lid_rad))
    return max(0.05, torque / L.LID_DAMP)


def lever_from_lid(lid_rad: float) -> float:
    if lid_rad <= 0.0:
        return 0.0
    return math.asin(min(1.0, (L.pin_rise(lid_rad) + L.TOP_GAP) / L.LEVER_REAR))


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.lever = 0.0
        self.lid = 0.0
        self.can = np.array([L.CAN_X, L.CAN_Y, L.TOP + L.CAN_H / 2])
        self.vz = 0.0
        self.grip: dict = {}  # arm -> can centre minus TCP when it was taken

    def _press(self, tcp, tips, opening) -> float:
        if opening >= 0.3:
            return 0.0
        bx, by = float(tips[0]) - L.BIN_X, float(tips[1]) - L.BIN_Y
        if not (L.PLATE_X0 - 0.004 <= bx <= L.PLATE_X1 and abs(by) <= L.PLATE_HY):
            return 0.0
        return L.lever_from_press(corner_z(tcp, tips) - L.TOP)

    def _rest_z(self) -> float:
        r = float(np.hypot(self.can[0] - L.BIN_X, self.can[1] - L.BIN_Y))
        if r <= L.IN_R - L.CAN_R + 0.004 and self.lid >= math.radians(30.0):
            return L.TOP + L.FLOOR_TOP + L.CAN_H / 2
        if r <= L.OUT_R + L.CAN_R / 2:
            return L.TOP + L.HINGE_Z + L.LID_T + L.CAP_H + L.CAN_H / 2
        return L.TOP + L.CAN_H / 2

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        press = 0.0
        if read is not None:
            for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
                press = max(press, self._press(tcp, tips, opening))
                tcp = np.asarray(tcp, dtype=float)
                holding = (opening * rigstate.FULL_GAP < HOLD_GAP and np.hypot(*(tcp[:2] - self.can[:2])) <= 0.02
                           and abs(float(tcp[2]) - self.can[2]) <= L.CAN_H / 2)
                if holding:
                    self.grip.setdefault(side, self.can - tcp)
                    self.vz = 0.0
                elif side in self.grip:
                    del self.grip[side]
            for side, off in self.grip.items():
                self.can = np.asarray(read[rigstate.ARMS.index(side)][0], dtype=float) + off
        lid_set = L.lid_from_rod(L.rod_from_lever(press))
        if lid_set >= self.lid:
            self.lid = lid_set
        else:
            self.lid = max(lid_set, self.lid - closing_rate(self.lid) * DT)
        self.lever = max(press, lever_from_lid(self.lid))
        if not self.grip:
            rest = self._rest_z()
            if self.can[2] > rest + 1e-9:
                self.vz += L.G * DT
                self.can = np.array([self.can[0], self.can[1], max(rest, self.can[2] - self.vz * DT)])
            else:
                self.vz = 0.0

    def states(self) -> dict:
        q = (0.0, 0.0, 0.0, 1.0)
        rise = L.rod_from_lever(self.lever)
        return {BIN: mk.state_from_poses(BIN_PART, (L.BIN_X, L.BIN_Y, L.TOP, *q),
                                         joints={"pedal": self.lever, "lid": self.lid}, source="offline"),
                ROD: mk.state_from_poses(ROD_PART, (L.BIN_X + L.ROD_X, L.BIN_Y, L.TOP + L.ROD_BOT + rise, *q),
                                         source="offline"),
                CAN: mk.state_from_poses(CAN_PART, (*self.can, *q), source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(BIN), "source", "sim") != "sim"


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
    """With TASK_TRACE=<file>, append one JSON line per control step: every part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The can in the bin, the lid never pushed, shut and still for 1 s, the bin kept, the can never on the lid; held
    1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline
    stand-in)."""
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
    try:
        row["rod_origin"] = parts[ROD].origin
        row["bin_origin"] = parts[BIN].origin
    except (AttributeError, KeyError, TypeError):
        row["rod_origin"] = row["bin_origin"] = None
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(BIN), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the bin tipped over 30 degrees or slid over 100 mm, or the can is not in the bin, or the lid was moved by
    anything but the pedal linkage (a finger zone within 1 mm of it; the lid rising over 5 degrees above what the rod
    and pedal give, or held 5 degrees above the rod's angle, or lifted by the can); 0.5 in the bin, never pushed, but
    not all of shut and still, bin kept, can never within 1 mm of the lid; 1.0 all of them. The latest scored step's
    score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm past the TCP along the approach axis. Both grips close along y (yaw 0); the left leans 60 degrees forward
# (fingers reaching forward and down round the can, the housing behind it), the right 25 degrees (its housing leans
# back, clear of the bin's front wall).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
CLOSED = 0.0
# 64 mm commanded across the 66 mm can: each finger 1 mm short of its target pushes 5000 N/m x 1 mm = 5 N (the sim's
# finger drive), about the real grip a thin empty can takes without denting (README: the can). Revision 19 Nov: was
# 60 mm (15 N a finger). 14 g needs well under 1 N of friction to hold.
CAN_GRIP = -(2 * L.CAN_R - 0.002) / 2
LET_GO = -(2 * L.CAN_R + 0.010) / 2  # 76 mm: wide enough to drop it, the fingers kept close to the can
PITCH_L, PITCH_R = math.radians(60.0), math.radians(25.0)
YAW_L = tiltgrip.tilted(0.0, PITCH_L)
YAW_R = tiltgrip.tilted(0.0, PITCH_R)
A_R = tiltgrip.rotation(0.0, PITCH_R)[:, 2]  # the right grip's approach axis

# The left TCP 90 mm over the table. With the 60-degree lean the pads meet the can's side from 74 to 106 mm up: on a
# real 330 ml can that is the top band of the straight wall, ending at the necked shoulder (about 103 mm up), where the
# neck and the seamed top stiffen it; not the middle of the wall (57 mm up), which dents most easily.
GRIP_Z = 0.090
CARRY_Z = GRIP_Z + L.RIM + 0.015  # the can's bottom 15 mm over the rim
# Over the opening, 32 mm in front of the axis (the can 6.6 mm inside the rim's inner edge): with the grip's
# finger zone (fingers held open by the can, 66 mm) (43 mm round the fingers) the zone clears the lid by 22 mm at 67 degrees, 5 mm at 60 and 1 mm at about
# 59; the can clears it by 33 mm at 67 degrees, 14 mm at 60.
DROP = (L.BIN_X - 0.032, L.BIN_Y + 0.005)
# Let go low: the can lowered 60 mm into the opening first (its bottom 45 mm under the rim, the fingers' lowest
# corner still about 27 mm over it), so it falls 153 mm instead of 218 and the open fingers start well under the lid.
RELEASE_Z = CARRY_Z - 0.060
UP_L = (L.BIN_X - 0.070, L.BIN_Y + 0.040, CARRY_Z + 0.030)  # back toward the arms and up, out of the lid's sweep
PARK_L = (0.200, 0.270, 0.300)

PRESS_OVER = 0.015  # the finger corner commanded 15 mm below where the plate's stop leaves it (the arm's give)
PRESS_XY = (L.BIN_X + L.PRESS_X, L.BIN_Y)
PARK_R = (0.150, -0.200, 0.150)


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _tips_at(d, x, y, corner, finger, label):
    """A right-arm waypoint that puts the finger's lowest corner at (x, y, corner) (the pad tips there plus the
    corner's drop)."""
    tip_z = corner + rigstate.FINGER_HALF_WIDTH * math.hypot(A_R[0], A_R[1])
    tcp = np.array([x, y, tip_z]) - PAD_TIP * A_R
    return _w(d, float(tcp[0]), float(tcp[1]), float(tcp[2]), finger, label, YAW_R)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def right_moves(press_corner=None, release_at=None, left=None):
    """The right arm: closed fingers over the pedal, down onto it, press it to its stop, hold it while the can goes
    in and the left hand is clear, rise slowly (the lid closes behind the hand), aside."""
    px, py = PRESS_XY
    full = L.PLATE_TOP - L.PEDAL_TRAVEL - PRESS_OVER
    pc = full if press_corner is None else press_corner
    plan = [
        _tips_at(1.5, px - 0.080, py - 0.060, L.PLATE_TOP + 0.090, CLOSED, "in front of the bin, fingers closed into a "
                 "block"),
        _tips_at(1.5, px, py, L.PLATE_TOP + 0.060, CLOSED, "over the pedal"),
        _tips_at(1.5, px, py, L.PLATE_TOP + 0.008, CLOSED, "down to just over the plate"),
        _tips_at(2.5, px, py, pc, CLOSED, "press the pedal to its stop: the lid opens"),
    ]
    hold_to = release_at if release_at is not None else _until(left, "back and up") + 0.5
    plan.append(_tips_at(hold_to - _duration(plan), px, py, pc, CLOSED, "hold the pedal down"))
    plan += [
        _tips_at(2.5, px, py, L.PLATE_TOP + 0.060, CLOSED, "rise slowly: the lid closes behind the hand"),
        _w(1.5, *PARK_R, CLOSED, "aside", YAW_R),
    ]
    return plan


def left_moves(drop=DROP):
    """The left arm: round the can, grip it, lift it, over the opening, lower it 60 mm into it, let go, straight up,
    back, aside."""
    cx, cy = L.CAN_X, L.CAN_Y
    return [
        _w(2.5, cx, cy, GRIP_Z + 0.090, OPEN, "over the can", YAW_L),
        _w(1.5, cx, cy, GRIP_Z, OPEN, "down round the can", YAW_L),
        _w(0.3, cx, cy, GRIP_Z, OPEN, "settle", YAW_L),
        _w(1.2, cx, cy, GRIP_Z, CAN_GRIP, "grip the can", YAW_L),
        _w(2.0, cx, cy, CARRY_Z, CAN_GRIP, "lift it over the rim's height", YAW_L),
        _w(2.5, *drop, CARRY_Z, CAN_GRIP, "over the opening, in front of the open lid", YAW_L),
        _w(1.2, *drop, RELEASE_Z, CAN_GRIP, "lower it into the opening", YAW_L),
        _w(0.5, *drop, RELEASE_Z, CAN_GRIP, "settle", YAW_L),
        _w(1.0, *drop, RELEASE_Z, LET_GO, "let go: the can drops in", YAW_L),
        _w(1.8, *UP_L, LET_GO, "back and up, out of the lid's way", YAW_L),
        _w(1.5, *PARK_L, LET_GO, "aside", YAW_L),
    ]


# Wrong ways (counterfactuals, run on the GPU for the README; the offline model predicts them roughly):
#   half    the pedal pressed only half way: the lid stops near 33 degrees and the can meets it
#   early   the pedal let go 0.2 s after the can is let go, before the left hand is back: the closing lid meets
#           the leaving hand
VARIANTS = ("half", "early")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    left = left_moves()
    if variant == "half":
        right = right_moves(press_corner=L.PLATE_TOP - L.PEDAL_TRAVEL / 2, left=left)
    elif variant == "early":
        right = right_moves(release_at=_until(left, "let go") + 0.2, left=left)
    else:
        right = right_moves(left=left)
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      half   the pedal pressed half way: the lid reaches about 33 degrees and the can hits it
      early  the pedal let go right after the can is let go: the lid closes onto the leaving hand"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
