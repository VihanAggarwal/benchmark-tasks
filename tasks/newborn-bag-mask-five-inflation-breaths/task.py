"""On the station table, real size: a newborn training manikin lies on its back along x on the midline, its crown
toward the robot (x 164 mm) and its face up (the mask's target point at x 250 mm, y 0, 90 mm over the table). In
front of its crown a fixed tray holds an infant self-inflating bag with a size-1 round face mask (64 mm across) on
its swivel elbow, mask down at x 110 mm, y 0, the bag toward -y.

Goal: five inflation breaths of 2 to 3 s each, every one raising the chest, the mask held sealed on the face all the
while, never on the eyes; at the end the mask still held sealed and the bag let go. The bag hangs from the valve on a
free bending neck and the mask turns freely on its swivel, so pressing the mask on the face and squeezing the bag
take two hands: one holds the valve elbow and presses the mask down, the other holds the bag level and squeezes it.

The bag and mask is one loose mechkit articulation (mechkit.py, next to this file): the mask's shell; the cushion as
8 pads on sprung slides; the valve on a free swivel; the bag on a free neck hinge with stops at +-0.5 rad; the bag's
two squeeze panels on sprung slides. The manikin's head (skull, a level face seat, nose, mouth, eyes) and the tray are
fixed; the manikin's body is fixed with its chest plate on a sprung vertical slide. Air is a declared rule
(scoring.Lung): every control step (``mk.on_step``) it reads the seal (pad compressions, tilt, place) and the squeeze
and drives the chest plate's target. The scorer runs the same rule. The arms are read through rigstate.py.

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
import tiltgrip  # noqa: E402  (the right's rolled grip: wraps yam_arena.demo.grip_rotation)
from scoring import BODY, BVM, HEAD, TRAY  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s at rest

INSTRUCTION = ("Give the newborn manikin five inflation breaths of 2 to 3 s each with the bag and mask from the "
               "tray, so that its chest rises with every breath. Do not press on its eyes. Finish with the mask "
               "still held on the face and the bag let go.")


# -- the parts --------------------------------------------------------------------------------------------------------
PAD_LINKS = tuple(mk.slider(f"pad_{i}", "Z", P.pad_shapes(i), upper=L.PAD_TRAVEL, anchor=P.pad_centre(i),
                            mass=L.PAD_MASS, spring=L.PAD_K, damping=L.PAD_DAMP, color=L.CLEAR_SILICONE)
                  for i in range(L.PAD_N))
VALVE = mk.hinge("valve", "Z", (0.0, 0.0, L.SWIVEL_Z), P.valve_shapes(), lower=-L.SWIVEL_LIMIT,
                 upper=L.SWIVEL_LIMIT, mass=L.VALVE_MASS, damping=L.SWIVEL_DAMP, color=L.VALVE_BLUE)
BAG = mk.hinge("bag", "X", (0.0, L.NECK_Y, L.HOUSING_Z), P.bag_shapes(), lower=-L.NECK_LIMIT, upper=L.NECK_LIMIT,
               mass=L.BAG_MASS, damping=L.NECK_DAMP, parent="valve", color=L.BAG_GREEN)
PANELS = (mk.slider("squeeze_a", "-X", P.panel_shapes(1), upper=L.PANEL_TRAVEL, mass=L.PANEL_MASS,
                    spring=L.PANEL_K, damping=L.PANEL_DAMP, parent="bag", color=L.BAG_GREEN),
          mk.slider("squeeze_b", "X", P.panel_shapes(-1), upper=L.PANEL_TRAVEL, mass=L.PANEL_MASS,
                    spring=L.PANEL_K, damping=L.PANEL_DAMP, parent="bag", color=L.BAG_GREEN))
BVM_PART = mk.assembly(BVM, origin=(L.BVM_X0, L.BVM_Y0, L.TOP + L.TRAY_Z), shapes=P.mask_shapes(),
                       mass=L.MASK_ROOT_MASS, joints=[*PAD_LINKS, VALVE, BAG, *PANELS], color=L.CLEAR_SILICONE,
                       friction=L.BVM_FRICTION, tight_fit=True)
CHEST = mk.slider("chest", "Z", P.chest_shapes(), upper=L.CHEST_TRAVEL, mass=L.CHEST_MASS, spring=L.CHEST_K,
                  damping=L.CHEST_DAMP, color=L.SKIN)
HEAD_PART = mk.assembly(HEAD, origin=(0.0, 0.0, L.TOP), shapes=P.head_shapes(), color=L.SKIN,
                        friction=L.FACE_FRICTION)
BODY_PART = mk.assembly(BODY, origin=(0.0, 0.0, L.TOP), shapes=P.body_shapes(), joints=[CHEST], color=L.SKIN)
TRAY_PART = mk.assembly(TRAY, origin=(0.0, 0.0, L.TOP), shapes=P.tray_shapes(), color=L.TRAY_COLOR)
OBJECTS = [HEAD_PART, BODY_PART, TRAY_PART, BVM_PART]
SPECS = {HEAD: HEAD_PART, BODY: BODY_PART, TRAY: TRAY_PART, BVM: BVM_PART}
FIXED = {HEAD: HEAD_PART, BODY: BODY_PART, TRAY: TRAY_PART}


# -- the air rule in the simulator ------------------------------------------------------------------------------------
LUNG = S.Lung()
CHEST_SAG = L.CHEST_MASS * L.G / L.CHEST_K  # the plate's own weight on its spring


@mk.on_step
def _air(states) -> None:
    """Every control step (sim): read the seal and the squeeze, run the air rule, drive the chest plate (targets go
    back to 0 at every reset, so it is set every step)."""
    r = S.read(states)
    if r is not None:
        LUNG.update(S.sealed(r) and not r.kinked, r.squeeze)
    mk.set_targets(BODY_PART, "chest", min(L.CHEST_TRAVEL, LUNG.rise + CHEST_SAG))


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows no hinges, pads or panels. Offline only (never in the simulator), this model gives the
# scorer: (1) the mask carried by a gripper closed under 23 mm with its TCP within 6 mm (top view) of the mask's
# axis and 8 mm of the elbow grip height, upright; (2) the bag held by a gripper closed under 74 mm with its TCP
# within 10 mm of the panels' centre: the squeeze is what the fingers' command leaves after the panels' springs
# (FINGER_SHARE); (3) both held: the neck's angle from the two hands' heights; the valve held alone: the bag droops to
# its stop (kinked); the bag held alone: the mask and valve hang from the neck at its stop, tilted 0.5 rad; (4) the
# mask pushed onto the face: the pads press by c = (k d + W) / (k + K) for an arm of k = 170 N/m commanded d under
# contact, W the mask's and valve's weight, K the cushion's 1200 N/m; (5) let go: the mask rests on the tray if over
# it, on the face if over the face's seat (the bag hanging off, too light to seal), else on the table. Rigid arms.
HOLD_VALVE = L.VALVE_W + 0.004
ARM_K = 170.0
W_MASK = (L.MASK_MASS + L.VALVE_MASS) * L.G
K_CUSHION = L.PAD_N * L.PAD_K
NECK_TO_PANEL = L.NECK_Y - L.PANEL_Y  # 88 mm
PANEL_REL = np.array([0.0, L.PANEL_Y, L.HOUSING_Z])  # the panels' centre in the mask frame at rest
HINGE_REL = np.array([0.0, L.NECK_Y, L.HOUSING_Z])
SEAT_LAT = L.FACE_R - L.RIM_OUT  # the whole cushion on the face's seat


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.origin = np.array([L.BVM_X0, L.BVM_Y0, L.TOP + L.TRAY_Z])
        self.tip = 0.0  # the mask frame turned about x (rad)
        self.neck = 0.0
        self.pad = 0.0
        self.squeeze = 0.0
        self.grip_l = None  # mask origin minus the left's TCP while carried
        self.grip_r = None  # panels' centre minus the right's TCP while held

    def panels(self) -> np.ndarray:
        """The panels' centre (world) as the model stands."""
        hinge = self.origin + rot_x(self.tip) @ HINGE_REL
        bag = rot_x(self.tip + self.neck)
        return hinge + bag @ (PANEL_REL - HINGE_REL)

    def _holds_valve(self, tcp, opening) -> bool:
        if opening * rigstate.FULL_GAP >= HOLD_VALVE:
            return False
        if self.grip_l is not None:
            return True
        tcp = np.asarray(tcp, dtype=float)
        return bool(abs(self.tip) < 1e-6
                    and math.hypot(tcp[0] - self.origin[0], tcp[1] - self.origin[1]) <= 0.006
                    and abs(float(tcp[2]) - (self.origin[2] + L.VALVE_GRIP_Z)) <= 0.008)

    def _holds_bag(self, tcp, opening) -> bool:
        if opening * rigstate.FULL_GAP >= L.BAG_ACROSS:
            return False
        if self.grip_r is not None:
            return True
        return bool(np.linalg.norm(np.asarray(tcp, dtype=float) - self.panels()) <= 0.010)

    @staticmethod
    def _support(xy) -> float:
        """The height (world) a mask set down at ``xy`` rests at: the tray, the face's seat or the table."""
        x, y = float(xy[0]), float(xy[1])
        if L.TRAY_X[0] <= x <= L.TRAY_X[1] and L.TRAY_Y[0] <= y <= L.TRAY_Y[1]:
            return L.TOP + L.TRAY_Z
        if math.hypot(x - L.FACE_X, y - L.FACE_Y) <= L.FACE_R:
            return L.TOP + L.FACE_Z
        return L.TOP

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        left = right = None
        if read is not None:
            left, right = read[rigstate.ARMS.index("left")], read[rigstate.ARMS.index("right")]
        on_r = right is not None and self._holds_bag(right[0], right[2])
        if on_r and self.grip_r is None:
            self.grip_r = self.panels() - np.asarray(right[0], dtype=float)
        if not on_r:
            self.grip_r = None
        on_l = left is not None and self._holds_valve(left[0], left[2])
        if on_l and self.grip_l is None:
            self.grip_l = self.origin - np.asarray(left[0], dtype=float)
        if not on_l:
            self.grip_l = None
        self.squeeze = 0.0
        if on_r:
            gap = right[2] * rigstate.FULL_GAP
            self.squeeze = float(np.clip((L.BAG_ACROSS - gap) * L.FINGER_SHARE, 0.0, 2 * L.PANEL_TRAVEL))
        self.pad = 0.0
        if on_l:
            want = np.asarray(left[0], dtype=float) + self.grip_l
            self.tip = 0.0
            support = self._support(want[:2])
            on_face = math.hypot(want[0] - L.FACE_X, want[1] - L.FACE_Y) <= SEAT_LAT
            if on_face and want[2] < support:
                d = support - float(want[2])
                self.pad = float(min(L.PAD_TRAVEL, (ARM_K * d + W_MASK) / (ARM_K + K_CUSHION)))
                want[2] = support - self.pad
            else:
                want[2] = max(float(want[2]), support)
            self.origin = want
            hinge_z = float(self.origin[2]) + L.HOUSING_Z
            if on_r:
                tcp_r = np.asarray(right[0], dtype=float) + self.grip_r
                s = (hinge_z - float(tcp_r[2])) / NECK_TO_PANEL
                self.neck = float(np.clip(math.asin(max(-1.0, min(1.0, s))), -L.NECK_LIMIT, L.NECK_LIMIT))
            else:
                # the bag lies level only with the mask set down off the face (the tray, the table); held up, or
                # on the face, nothing holds the bag up: it hangs at its stop
                resting = not on_face and self.origin[2] <= support + 0.002
                self.neck = 0.0 if resting else L.NECK_LIMIT
        elif on_r:
            panel = np.asarray(right[0], dtype=float) + self.grip_r
            lifted = panel[2] > self._support(panel[:2]) + L.HOUSING_Z + 0.005
            if lifted:  # the mask and valve hang from the neck at its stop, the bag level
                self.tip, self.neck = -L.NECK_LIMIT, L.NECK_LIMIT
            else:
                self.tip, self.neck = 0.0, 0.0
            hinge = panel - (PANEL_REL - HINGE_REL)
            origin = hinge - rot_x(self.tip) @ HINGE_REL
            origin[2] = max(float(origin[2]), self._support(origin[:2]))
            self.origin = origin
        else:
            self.tip = 0.0
            support = self._support(self.origin[:2])
            self.origin = np.array([self.origin[0], self.origin[1], support])
            if support == L.TOP + L.FACE_Z and math.hypot(self.origin[0] - L.FACE_X,
                                                          self.origin[1] - L.FACE_Y) <= SEAT_LAT:
                self.neck = L.NECK_LIMIT
                self.pad = (W_MASK + (L.BAG_MASS + 2 * L.PANEL_MASS) * L.G) / K_CUSHION
                self.origin[2] -= self.pad
            else:
                self.neck = 0.0

    def states(self) -> dict:
        joints = {f"pad_{i}": self.pad for i in range(L.PAD_N)}
        joints.update(valve=0.0, bag=self.neck, squeeze_a=self.squeeze / 2, squeeze_b=self.squeeze / 2)
        q = (math.sin(self.tip / 2), 0.0, 0.0, math.cos(self.tip / 2))
        return {BVM: mk.state_from_poses(BVM_PART, (*self.origin, *q), joints=joints, source="offline"),
                HEAD: HEAD_PART.rest_state(), BODY: BODY_PART.rest_state(), TRAY: TRAY_PART.rest_state()}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(BVM), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def _reset_all() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    LUNG.reset()


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
    """Exactly five good inflations (2 to 3 s each), nothing on the eyes, never over-inflated, dropped, held too long
    or lifted off, and for the last 1 s the mask sealed on the face, the bag let go and the chest down; held 1 s.
    This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline
    stand-in)."""
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
           "fallback_resets": _TRACKER.resets, "sim_rise_mm": round(LUNG.rise * 1000.0, 3), **result.details}
    try:
        row["bvm_joints"] = {k: round(float(v), 5) for k, v in parts[BVM].joints.items()}
        row["chest_mm"] = round(float(parts[BODY].joints.get("chest", 0.0)) * 1000.0, 3)
    except (AttributeError, KeyError, TypeError, ValueError):
        row["bvm_joints"] = None
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(BVM), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if an eye was touched or no inflation was good; 0.25 if over-inflated or dropped; 0.5 if an inflation was
    held too long; then by good inflations: 1.0 for exactly five (no others) with the mask sealed and the bag let go
    at the end and never lifted off, 0.75 for five or more otherwise, 0.5 for 3 or 4, 0.25 for 1 or 2. The latest
    scored step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points, heights above
# the table; the pad tips 13 mm past the TCP along the approach axis. The left grips straight down (yaw 0: fingers
# closing along y) and pinches the valve elbow over the housing, its pad tips 2 mm over the housing. The right (yaw
# 90 degrees: fingers closing along x) is rolled 40 degrees about x so that its approach axis leans toward +y: its
# wrist stays out toward -y, clear of the left's (straight down, the two wrists' links would overlap at 148 mm
# apart). It holds the bag across its two panels at its middle. Both lift together, carry the mask over the face,
# the left sets it down and presses, the right keeps the bag level at the neck's height and squeezes it five times.
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
ROLL_R = math.radians(40.0)
YAW_L = 0.0  # rad
YAW_R = tiltgrip.tilted(math.pi / 2, 0.0, ROLL_R)
A_R = tiltgrip.rotation(math.pi / 2, 0.0, ROLL_R)[:, 2]  # the right's approach axis: (0, 0.64, -0.77)
VALVE_GRIP = -(L.VALVE_W - 0.008) / 2  # 11 mm commanded across the 19 mm elbow: about 20 N a finger


def bag_finger(squeeze: float) -> float:
    """The finger command that leaves ``squeeze`` (both panels' travel added) once the panels' springs push back."""
    return -(L.BAG_ACROSS - squeeze / L.FINGER_SHARE) / 2


CARRY_SQUEEZE = 0.0065  # held through the carry and between breaths: a firm hold that gives no breath
BREATH_SQUEEZE = 0.0165  # 10 mm more: 6 mm of chest rise by the air rule
BAG_CARRY = bag_finger(CARRY_SQUEEZE)
BAG_BREATH = bag_finger(BREATH_SQUEEZE)
LIFT_Z = L.FACE_Z + 0.022  # the cushion carried 22 mm over the face's height
PRESS_OVER = 0.019  # the left commanded 19 mm under contact: 2.7 to 8.0 N on the cushion for arms of 120 to 600 N/m
CONTACT_TCP = L.FACE_Z + L.VALVE_GRIP_Z
NEAR_TCP = CONTACT_TCP + 0.003
PRESS_TCP = CONTACT_TCP - PRESS_OVER
BAG_AT_FACE = L.FACE_Z - 0.003 + L.HOUSING_Z  # the bag's axis at the neck's height with the cushion pressed 3 mm
VALVE_AT = np.array([L.BVM_X0, L.BVM_Y0])
BAG_AT = np.array([L.BVM_X0, L.BVM_Y0 + L.PANEL_Y])
FACE_AT = np.array([L.FACE_X, L.FACE_Y])
BAG_FACE_AT = FACE_AT + np.array([0.0, L.PANEL_Y])
GRAB_L = L.TRAY_Z + L.VALVE_GRIP_Z
GRAB_R = L.TRAY_Z + L.HOUSING_Z
LIFT_DZ = LIFT_Z - L.TRAY_Z
SQUEEZE_S, HOLD_S, EASE_S, WAIT_S = 0.6, 2.0, 0.6, 1.0
BACK_R = 0.080  # out along the approach axis after letting go
PARK_R = np.array([0.150, -0.190, 0.170])  # aside, toward the robot: every finger zone 26 mm or more off the bag


def _w(d, p, finger, label, yaw):
    x, y, z = (float(v) for v in p)
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _xy(xy, z):
    return np.array([xy[0], xy[1], z])


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def right_moves(breaths: int = L.N_BREATHS, hold: float = HOLD_S, keep: bool = False):
    """The right arm: over the bag, fingers either side of its middle, hold it, lift it with the left, carry it
    beside the face, keep it level at the neck's height, squeeze it ``breaths`` times (each held ``hold`` s), let
    go, up, aside."""
    plan = [
        _w(2.0, _xy(BAG_AT, GRAB_R) - 0.070 * A_R, OPEN, "behind and over the bag", YAW_R),
        _w(1.2, _xy(BAG_AT, GRAB_R), OPEN, "fingers either side of the bag's middle", YAW_R),
        _w(0.3, _xy(BAG_AT, GRAB_R), OPEN, "settle", YAW_R),
        _w(1.5, _xy(BAG_AT, GRAB_R), BAG_CARRY, "hold the bag", YAW_R),
        _w(0.3, _xy(BAG_AT, GRAB_R), BAG_CARRY, "settle", YAW_R),
        _w(1.5, _xy(BAG_AT, GRAB_R + LIFT_DZ), BAG_CARRY, "lift it with the mask", YAW_R),
        _w(2.0, _xy(BAG_FACE_AT, GRAB_R + LIFT_DZ), BAG_CARRY, "carry it beside the face", YAW_R),
        _w(1.0, _xy(BAG_FACE_AT, GRAB_R + LIFT_DZ - (LIFT_Z + L.VALVE_GRIP_Z - NEAR_TCP)), BAG_CARRY,
           "down with the mask", YAW_R),
        _w(1.0, _xy(BAG_FACE_AT, BAG_AT_FACE), BAG_CARRY, "level at the neck's height", YAW_R),
        _w(0.5, _xy(BAG_FACE_AT, BAG_AT_FACE), BAG_CARRY, "settle", YAW_R),
    ]
    at = _xy(BAG_FACE_AT, BAG_AT_FACE)
    for n in range(1, breaths + 1):
        plan += [
            _w(SQUEEZE_S, at, BAG_BREATH, f"breath {n}: squeeze", YAW_R),
            _w(hold, at, BAG_BREATH, f"breath {n}: hold the squeeze", YAW_R),
            _w(EASE_S, at, BAG_CARRY, f"breath {n}: ease off", YAW_R),
            _w(WAIT_S, at, BAG_CARRY, f"breath {n}: let the chest fall", YAW_R),
        ]
    if keep:
        plan.append(_w(3.4, at, BAG_CARRY, "keep holding the bag", YAW_R))
    else:
        plan += [
            _w(1.2, at, OPEN, "let go of the bag", YAW_R),
            _w(1.0, at - BACK_R * A_R, OPEN, "back out", YAW_R),
            _w(1.2, PARK_R, OPEN, "aside", YAW_R),
        ]
    return plan


def left_moves(right, light: bool = False):
    """The left arm: over the valve, fingers either side of the elbow, pinch it, lift the mask with the right, carry
    it over the face, set it down on the target point, press it on and hold it there to the end."""
    press = NEAR_TCP if light else PRESS_TCP
    plan = [
        _w(2.0, _xy(VALVE_AT, GRAB_L + 0.057), OPEN, "over the valve elbow", YAW_L),
        _w(1.2, _xy(VALVE_AT, GRAB_L), OPEN, "fingers either side of the elbow", YAW_L),
        _w(0.3, _xy(VALVE_AT, GRAB_L), OPEN, "settle", YAW_L),
        _w(1.5, _xy(VALVE_AT, GRAB_L), VALVE_GRIP, "pinch the elbow", YAW_L),
        _w(0.3, _xy(VALVE_AT, GRAB_L), VALVE_GRIP, "settle", YAW_L),
        _w(1.5, _xy(VALVE_AT, LIFT_Z + L.VALVE_GRIP_Z), VALVE_GRIP, "lift the mask with the bag", YAW_L),
        _w(2.0, _xy(FACE_AT, LIFT_Z + L.VALVE_GRIP_Z), VALVE_GRIP, "over the face", YAW_L),
        _w(1.0, _xy(FACE_AT, NEAR_TCP), VALVE_GRIP, "down onto the face", YAW_L),
        _w(1.0, _xy(FACE_AT, press), VALVE_GRIP, "press the mask on", YAW_L),
        _w(0.5, _xy(FACE_AT, press), VALVE_GRIP, "settle", YAW_L),
    ]
    plan.append(_w(_duration(right) + 1.0 - _duration(plan), _xy(FACE_AT, press), VALVE_GRIP,
                   "hold the mask sealed", YAW_L))
    return plan


# Wrong ways (counterfactuals, run on the GPU for the README; the offline model predicts them roughly):
#   four   only four breaths: 0.5
#   short  each squeeze held 1.2 s instead of 2.0 s: the inflations are too short to count, 0
#   light  the mask set on the face but not pressed: it leaks, no breath counts, 0
#   keep   the right keeps hold of the bag at the end: 0.75
VARIANTS = ("four", "short", "light", "keep")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = right_moves(breaths=4 if variant == "four" else L.N_BREATHS,
                        hold=1.2 if variant == "short" else HOLD_S, keep=variant == "keep")
    return {"left": left_moves(right, light=variant == "light"), "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo): four, short, light, keep."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
