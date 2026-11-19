"""On the station table, real size: a stack of three knocked-down 12 x 8 x 6 in cartons (508 x 356 x 6.4 mm each, flaps
out) on a fixed brown board at x 270 mm, centred on y 0. On top of the stack lies an empty 12 x 8 x 6 in regular
slotted carton (RSC) in 3.2 mm single-wall C-flute on one long side with its open ends along x: a 305 (y) x 152 (x) x
203 mm tall four-panel sleeve with a 102 mm flap on each panel at each open end. It was emptied through its near (-x)
end, whose four flaps lie out straight; the tape on its far (+x) end has just been slit and those four flaps hang
loose (top one drooping 45 degrees, bottom one 30 degrees up, the -y end panel's 30 degrees out past straight, the
+y one's straight; layout.FLAP_SET). Its corners are creases (rotating joints about axes along x); the fourth corner
is the manufacturer's glue joint, a 38 mm tab creased on the end panel's edge and glued (a mechkit bond, a FixedJoint
that never breaks) to the underneath panel, so the sleeve is a closed four-bar. Each flap is a rigid plate on its own
crease, drawn at its start angle: a spring holds it at its set (at first the start angle), and bent more than 3 degrees
from that set the crease takes a new set where the flap is (a declared rule, below), so a flap stays where it is put.

Goal: the carton collapsed flat (every corner crease within 15 degrees of shut or open, every flap within 15 degrees
of straight out, 12 mm thick at most) lying on the stack, the stack still as found, hands off. Racking the sleeve
toward +y by pushing its -y end panel turns the four corner creases together; past about 45 degrees its own weight
finishes the job, and the stack's top folds the drooping top flap out as the top panel comes down. Presses out past
the far crease fold the raised far flaps out flat (both arms, last), and presses along the top seat the layers.

The carton's panels and the stack's flats are mechkit compounds (mechkit.py, next to this file). The arms are read
through rigstate.py (the let-go check). One declared rule: the flap creases' set (``crease_set_rule``, sim only), which
moves each flap's spring rest (its drive target) to the flap's angle once the flap is bent past the yield angle.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=shallow, DEMO_VARIANT=flaps_left.  Flap models (the submitted demo's plan, another flap crease model; read
at import, layout.FLAP_MODELS): DEMO_VARIANT=crease_set (the submitted model), crease_stiff, crease_ratchet,
friction_hold (the 18 Nov model).  Trace: TASK_TRACE=<file> (one JSON line per control step).
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
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import CARTON, FLATS  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Collapse the empty carton flat where it lies on the stack, and leave the stack in place."

DEMO_VARIANTS = ("shallow", "flaps_left")  # counterfactual plans (the submitted flap model)
MODEL_VARIANTS = tuple(L.FLAP_MODELS)  # the submitted plan with another flap crease model


def flap_model_name(variant: str | None = None) -> str:
    """The flap crease model a DEMO_VARIANT selects: a name of layout.FLAP_MODELS, else the submitted one."""
    v = os.environ.get("DEMO_VARIANT", "") if variant is None else variant
    return v if v in L.FLAP_MODELS else L.FLAP_MODEL_DEFAULT


FLAP_MODEL_NAME = flap_model_name()  # read once at import: the joints are built from it
FLAP_MODEL = L.FLAP_MODELS[FLAP_MODEL_NAME]


# -- the parts --------------------------------------------------------------------------------------------------------
def _panel(body):
    centre, size, _ = L.PANELS[body]
    return mk.box(centre, size, L.CARTON_BROWN)


def _crease(body):
    return mk.hinge(body, "X", L.ANCHOR[body], [_panel(body)], upper=L.CREASE_LIMIT, lower=-L.CREASE_LIMIT,
                    mass=L.PANELS[body][2], damping=L.CREASE_DAMPING, friction=L.CREASE_FRICTION,
                    parent=L.PARENT[body])


def _flap(body, model=None):
    """A flap: a rigid board plate on a crease along its panel's open-end edge, drawn at its start angle (its first
    set, where its crease spring rests); ``crease_set_rule`` moves the set as the flap is bent; it turns about 97
    degrees either side of straight out."""
    model = FLAP_MODEL if model is None else model
    centre, size, mass = L.PANELS[body]
    a = np.asarray(L.ANCHOR[body], dtype=float)
    r = S.set_rot(body)
    shape = mk.box(tuple(a + r @ (np.asarray(centre) - a)), size, L.CARTON_BROWN, quat=tuple(mk.matrix_to_quat(r)))
    s = L.FLAP_SET[body]
    return mk.hinge(body, L.AXIS[body], L.ANCHOR[body], [shape], upper=L.FLAP_LIMIT - s, lower=-L.FLAP_LIMIT - s,
                    mass=mass, spring=model["spring"], damping=model["damping"], friction=model["friction"],
                    parent=L.PARENT[body])


BOARD = mk.assembly("board", origin=(L.PILE_X, L.PILE_Y, L.TOP),
                    shapes=[mk.box((0.0, 0.0, L.BOARD_T / 2), (L.BOARD_X, L.BOARD_Y, L.BOARD_T), L.BOARD_BROWN)],
                    color=L.BOARD_BROWN, friction=L.FLAT_FRICTION, tight_fit=True)
FLAT_PARTS = [mk.assembly(FLATS[i], origin=(L.PILE_X, L.PILE_Y, L.TOP + L.flat_z(i)), mass=L.FLAT_MASS,
                          shapes=[mk.box((0.0, 0.0, 0.0), (L.FLAT_X, L.FLAT_Y, L.FLAT_T), L.PILE_BROWN)],
                          color=L.PILE_BROWN, friction=L.FLAT_FRICTION, tight_fit=True) for i in range(L.FLAT_N)]
CARTON_PART = mk.assembly(CARTON, origin=(L.CARTON_X, L.CARTON_Y, L.TOP + L.CARTON_ORIGIN_Z), mass=L.PANELS["base"][2],
                          shapes=[_panel("base")],
                          joints=[*(_crease(b) for b in L.CREASES), *(_flap(b) for b in L.FLAPS)],
                          color=L.CARTON_BROWN, friction=L.CARTON_FRICTION, tight_fit=True)
SEAM = mk.bond("seam_glue", (CARTON_PART, "tab"), CARTON_PART)  # the glue joint: the tab glued to A
OBJECTS = [BOARD, *FLAT_PARTS, CARTON_PART]
SPECS = {CARTON: CARTON_PART, **{FLATS[i]: FLAT_PARTS[i] for i in range(L.FLAT_N)}}


# -- the flap creases' set (the declared rule; sim only) --------------------------------------------------------------
# A scored crease in corrugated board is springy for a few degrees about where it rests and, bent further, yields and
# rests where it was left: a flap opened out stays open, a flap pushed flat stays flat. In the sim each flap's spring
# is a joint drive toward its set (the drive's target, 0 = the start angle as drawn); at the start of every control
# step this rule moves the set to the flap's angle once the flap is more than the yield angle from it. Between those
# moves the spring holds the flap at the position level (a drive, not joint friction, which crept on the GPU).
_SETS: dict = {}  # flap -> its set now (joint position, rad); empty after every reset (every set as drawn, 0)


def next_set(k: str, q: float, t: float, model=None) -> float:
    """Flap ``k``'s set (joint rad) after a step at joint position ``q`` with set ``t``: ``q`` once ``q`` is more than
    the yield angle from ``t`` (for a ratchet model only when that brings the flap nearer straight), else ``t``."""
    model = FLAP_MODEL if model is None else model
    if model["yield_deg"] is None or not (math.isfinite(q) and math.isfinite(t)):
        return t
    if abs(q - t) <= math.radians(model["yield_deg"]):
        return t
    if model["ratchet"] and abs(L.flap_off(k, q)) >= abs(L.flap_off(k, t)):
        return t
    return q


@mk.on_step
def crease_set_rule(states) -> None:
    """In the simulator, at the start of every control step: each flap's set follows it once bent past the yield."""
    if FLAP_MODEL["yield_deg"] is None:
        return
    c = states.get(CARTON)
    joints = getattr(c, "joints", None)
    if not joints:
        return
    for j in CARTON_PART.joints:
        if j.name not in L.FLAPS:
            continue
        try:
            q = float(joints[j.name])
        except (KeyError, TypeError, ValueError):
            continue
        t = min(j.upper, max(j.lower, next_set(j.name, q, _SETS.get(j.name, 0.0))))
        _SETS[j.name] = t
        mk.set_targets(CARTON_PART, j.name, t)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it knows nothing of creases. Offline
# only (never in the simulator), this model gives the scorer the carton racked toward +y by one angle PHI (the rack
# of the four-bar sleeve; see layout.py): a closed gripper whose pad face pushes the -y end panel's outer face at a
# height ``hz`` over the stack's top, moved ``dy`` past the panel's foot, leans it to atan(dy / hz); once the lean
# passes GRAVITY_PHI the carton's own weight carries it on at 1.2 rad/s to flat. PHI never decreases.
# The far end's loose flaps (layout.FLAP_SET): the top panel's drooping flap is folded out by the stack's top as the
# carton comes down (its tip may not go below the stack's top); a raised flap is pressed down by a closed block whose
# pad tips are over it (its upper face may not stand above the pad tips), but only if the pad comes down inside the
# flap's reach (a flap standing steep has its tip nearer the crease than the pad, and the pad misses it). The
# underneath panel's raised flap can be pressed at any time, the end panels' flaps only on the flat carton. The stack
# never moves. The flap creases follow the task's flap model (FLAP_MODEL): with a crease set (the submitted model) a
# flap not held by a pad or the stack springs back to its set at once, and its set follows it as crease_set_rule
# does (more than the yield angle away: the set moves there), so a pressed flap ends up to the yield angle back
# toward where it came from (its weight's sag, which lowers it, is left out). With the 18 Nov model (friction_hold) a
# flap stays where it is put, except that once the carton is flat the lower layer's two far flaps (the underneath
# panel's and the +y end panel's, which lie on the stack) creep tip up at the 18 Nov GPU trace's rates (CREEP_DEG_S).
PAD_FACE = 0.0059  # a closed pad block's face ahead of the TCP along the closing direction (GPU: 5.9 mm, shim task)
D_FACE_Y = L.A_YMIN - L.CT / 2  # the -y end panel's outer face at its foot
GRAVITY_PHI = 0.80  # rad (46 degrees): past this the weight finishes the rack (offline assumption; see README)
FALL_RATE = 1.2  # rad/s
HZ_MIN, HZ_MAX = 0.04, 0.19  # the pad block touches the end panel between these heights over the stack's top
X_CREASE_W = L.CARTON_X + L.X_FAR  # the far flaps' creases (world x, 346.2 mm)
FLAP_Y = {"af": (L.A_YMIN, L.A_YMAX), "bf": (L.A_YMAX, L.A_YMAX + L.WB),  # their y span once the carton is flat
          "df": (L.A_YMIN, L.A_YMIN + L.WB)}
FLAP_TOP = {"af": L.PT + L.CT, "bf": L.PT + L.CT, "df": L.PT + 2 * L.CT}  # their upper face lying straight
# GPU 18 Nov (trace_sampled_demo): once flat, af rose 0.09 to 0.14 degrees a step (8.4 to 63 degrees after its press,
# whatever its angle), bf 0.03 to 0.065; the rates here are the high ends. Both turn tip up (negative, as their set).
CREEP_DEG_S = {"af": 2.8, "bf": 1.3}  # the friction_hold model only: a drive holds the crease models' flaps
LATE_PRESS = ("bf", "df")  # on the flat carton only


def top_flap_droop_max(phi: float) -> float:
    """The most the top panel's far flap can droop (rad) at rack ``phi`` with its tip no lower than the stack's top."""
    frames = CARTON_PART.link_transforms(L.joints_for(phi))
    h = float(frames["cf"][1][2]) + L.CARTON_ORIGIN_Z - L.PT  # its crease over the stack's top
    return math.asin(max(0.0, min(1.0, (h - L.CT / 2) / L.FD)))


class OfflineModel:
    def __init__(self, model=None) -> None:
        self.model = FLAP_MODEL if model is None else model
        self.reset()

    def reset(self) -> None:
        self.phi = 0.0
        self.off = dict(L.FLAP_SET)  # each flap's angle from straight (rad, signed as FLAP_SET)
        self.set = dict(L.FLAP_SET)  # each flap's set, the same way (the crease models)

    def _follow_sets(self) -> None:
        """crease_set_rule in angles from straight (a joint differs from it by the start angle only)."""
        for k in L.FLAPS:
            q, t = L.flap_joint(k, self.off[k]), L.flap_joint(k, self.set[k])
            self.set[k] = L.flap_off(k, next_set(k, q, t, self.model))

    def _press(self, k: str, tips) -> None:
        x, y, z = (float(v) for v in tips)
        lo, hi = FLAP_Y[k]
        r = x - X_CREASE_W
        if not (0.005 <= r <= L.FD * math.cos(self.off[k]) and lo <= y <= hi):
            return
        allowed = math.atan2(max(0.0, z - L.TOP - FLAP_TOP[k]), r)
        if allowed < abs(self.off[k]):
            self.off[k] = math.copysign(allowed, self.off[k])

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        flat = self.phi >= math.pi / 2 - 1e-9
        creases = self.model["yield_deg"] is not None
        if creases:  # unheld, each flap springs back to its set; a pad or the stack below holds it again
            self.off = dict(self.set)
        elif flat:
            for k, rate in CREEP_DEG_S.items():
                self.off[k] = max(-L.FLAP_LIMIT, self.off[k] - math.radians(rate) / L.CONTROL_HZ)
        if read is not None:
            for tcp, tips, opening in read:
                if opening > 0.3:
                    continue
                for k in FLAP_Y:
                    if flat or k not in LATE_PRESS:
                        self._press(k, tips)
                hz = float(tcp[2]) - L.TOP - L.PT
                if not (HZ_MIN <= hz <= HZ_MAX):
                    continue
                if abs(float(tcp[0]) - L.CARTON_X) > L.LX / 2 + L.FD:  # the end panel or a flap in its plane
                    continue
                dy = float(tcp[1]) + PAD_FACE - D_FACE_Y
                if dy <= 0.0 or float(tcp[1]) > L.A_YMAX:
                    continue
                self.phi = max(self.phi, min(math.pi / 2, math.atan2(dy, hz)))
        if GRAVITY_PHI <= self.phi < math.pi / 2:
            self.phi = min(math.pi / 2, self.phi + FALL_RATE / L.CONTROL_HZ)
        self.off["cf"] = min(self.off["cf"], top_flap_droop_max(self.phi))
        if creases:
            self._follow_sets()

    def joints(self) -> dict:
        return L.joints_for(self.phi, flaps=self.off)

    def states(self) -> dict:
        out = {CARTON: mk.state_from_poses(CARTON_PART, (L.CARTON_X, L.CARTON_Y, L.TOP + L.CARTON_ORIGIN_Z,
                                                         0.0, 0.0, 0.0, 1.0), joints=self.joints(),
                                           source="offline")}
        for n in FLATS:
            out[n] = SPECS[n].rest_state()
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(CARTON), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _SETS.clear()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _SETS.clear()  # mechkit has put every drive target back to 0 (the start angles)


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the carton's creases and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The carton flat on the stack, the stack kept, both grippers 30 mm clear and everything still for 1 s, no
    latched knock; held 1 s. This runs first every control step, so it also feeds the tracker (and, without a
    simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
            _SETS.clear()
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
        row["creases_rad"] = {k: round(float(parts[CARTON].joints[k]), 4) for k in L.JOINTS}
    except (AttributeError, KeyError, TypeError, ValueError):
        row["creases_rad"] = None
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    try:
        row["seam_glue"] = SEAM.status()
    except Exception:  # noqa: BLE001 - a trace field only (a declaration offline)
        row["seam_glue"] = None
    row["flap_model"] = FLAP_MODEL_NAME
    if offline:
        row["flap_sets_deg"] = {k: round(math.degrees(v), 1) for k, v in _OFFLINE.set.items()}
    elif FLAP_MODEL["yield_deg"] is not None:  # each set as an angle from straight, as flaps_deg
        row["flap_sets_deg"] = {k: round(math.degrees(L.flap_off(k, _SETS.get(k, 0.0))), 1) for k in L.FLAPS}
    row["source"] = getattr(parts.get(CARTON), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any flat of the stack was moved over 40 mm or turned over 30 degrees, or unless every part is let go and
    still (no panel, flap or flat moving 5 mm or turning 5 degrees over the last 1 s); 1.0 the carton flat on the stack
    with the stack kept; 0.75 flat on it but offset or leaning; 0.5 flat anywhere else; 0.25 not flat but no more
    than 110 mm tall; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Gripper closing axis along y (yaw 0), wrist straight down. Finger targets: the joint value is minus half
# the commanded gap. Every contact is the closed pad block's face (no grasp): the push needs well under 1 N (four
# 0.01 N m corner creases over a 0.12 m lever, plus damping), the presses only the arm's own stiffness. The push and
# the presses along the top work on the carton's centre line (x 270 mm, the panels, 76 mm short of every flap). The
# flap presses (written for the 18 Nov flap model, whose far flaps crept up once flat, af about 2.4 degrees a second,
# bf about 1; kept with the crease set as margin): the right arm presses the underneath panel's flap (and the -y end
# panel's above it) 49 mm out from the crease straight after the rack, where it still catches that flap standing up
# to 61 degrees, and both arms press the far flaps together as the very last contact, 70 mm out (af by the right
# arm, the +y end panel's bf, out of the right arm's reach, by the left), then lift 0.6 s, 3.5 s before the last step.
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
CLOSED = 0.0
YAW = 0.0
XC = L.CARTON_X

PUSH_HZ = 0.12  # the pad block pushes the end panel 120 mm over the stack's top (of its 203 mm)
PUSH_PHI = 0.95  # rad: pushed until the panel leans 54 degrees (past GRAVITY_PHI; the weight takes it on)
PUSH_Y0 = D_FACE_Y - PAD_FACE - 0.012  # TCP y of the closed block, 12 mm short of the panel's foot (in the air)
PUSH_Y1 = D_FACE_Y - PAD_FACE + PUSH_HZ * math.tan(PUSH_PHI)  # TCP y at the end of the push
PUSH_SPEED = 0.030  # m/s
TRAVEL_Z = 0.17  # TCP height over the table while moving about
PUSH_HOVER = 0.20  # TCP height of the closed block coming over to the end panel's foot (outside the carton)
PRESS_Z = L.PT + 2 * L.CT + PAD_TIP - 0.001  # pad tips 1 mm into the flat carton's top (6.4 mm over the stack's top)
HOVER_PRESS = L.PT + 0.10
PRESS_R = (-0.02, -0.14)  # the right arm's two press points along y (its own side, up to the midline)
PRESS_L = (0.24, 0.12)  # the left arm's, at the same times as the right arm's, 260 mm away
FLAP_PRESS_X = X_CREASE_W + 0.049  # 395 mm: the first flap press, 49 mm out (catches af standing up to 61 degrees)
FLAP_FINAL_X = X_CREASE_W + 0.070  # 416 mm: the last presses, 70 mm out (catch a flap up to 46 degrees, fold finer)
FLAP_PRESS_Y = -0.15  # right arm: over the underneath panel's flap and the -y end panel's (y -254 to -51 mm flat)
FLAP_PRESS_YL = 0.15  # left arm: over the +y end panel's flap (y 51 to 254 mm flat) and the top panel's
FLAP_PRESS_Z = L.PT + 2 * L.CT + PAD_TIP - 0.002  # pad tips 2 mm into the flat carton's top, over the flaps
FLAP_HOVER = 0.135  # flap presses come in from 122 mm (pad tips) over the table (reach: 145 mm at x 416)
FINAL_UP_S = 0.6  # the last lift: quick, so the far flaps creep as little as possible before the last step
SETTLE_S = 3.0  # wait for the weight to finish the rack
PARK_R = (0.10, -0.36, 0.17)
PARK_L = (0.10, 0.36, 0.17)


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, YAW)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def push_moves(phi=PUSH_PHI):
    """The right arm: closed block in the air 12 mm short of the end panel's foot, lowered to the push height, pushed
    in +y until the panel leans ``phi``, held while the weight finishes the rack, lifted clear."""
    y1 = D_FACE_Y - PAD_FACE + PUSH_HZ * math.tan(phi)
    z = L.PT + PUSH_HZ
    return [
        _w(2.5, XC, PUSH_Y0, PUSH_HOVER, CLOSED, "close the fingers into a block, over the air short of the foot"),
        _w(2.0, XC, PUSH_Y0, z + 0.03, CLOSED, "straight down outside the end panel (no open finger over the carton)"),
        _w(1.0, XC, PUSH_Y0, z, CLOSED, "down to the push height"),
        _w(abs(y1 - PUSH_Y0) / PUSH_SPEED, XC, y1, z, CLOSED, "push the end panel over toward +y"),
        _w(SETTLE_S, XC, y1, z, CLOSED, "hold while its own weight finishes the rack"),
        _w(1.0, XC, y1 - 0.01, TRAVEL_Z, CLOSED, "straight up, clear"),
    ]


def press_moves(ys, x=XC, what=""):
    """Closed block pressed 1 mm into the flat carton's top at each y (at ``x``), then lifted."""
    plan = []
    for y in ys:
        plan += [
            _w(1.5, x, y, HOVER_PRESS, CLOSED, f"over y {y * 1000:.0f} mm{what}"),
            _w(1.2, x, y, PRESS_Z, CLOSED, f"press down at y {y * 1000:.0f} mm{what}"),
            _w(0.5, x, y, PRESS_Z, CLOSED, "hold the press"),
            _w(1.0, x, y, HOVER_PRESS, CLOSED, "up"),
        ]
    return plan


def flap_press(x, y, what, up_s=1.0, up_z=FLAP_HOVER):
    """Closed block brought in high over a raised far flap, pressed 2 mm into the flat top at (x, y), then lifted."""
    return [
        _w(1.5, x, y, FLAP_HOVER, CLOSED, f"over x {x * 1000:.0f}, y {y * 1000:.0f} mm, {what}"),
        _w(1.5, x, y, FLAP_PRESS_Z, CLOSED, f"press down on {what}"),
        _w(0.5, x, y, FLAP_PRESS_Z, CLOSED, "hold the flap press"),
        _w(up_s, x, y, up_z, CLOSED, "up off the flaps"),
    ]


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the right arm pushes the end panel over (the weight finishes it, and the stack's top folds the
    drooping top flap out), lifts clear and presses the underneath panel's raised far flap (and the -y end panel's)
    down; then both arms press along the top at the same times (right at y -20 and -140 mm, left at 240 and 120 mm);
    last, both press the far flaps together (right: the underneath and -y end panels' flaps; left: the +y end and top
    panels'), lift 0.6 s and hold there for the policy's 3 s."""
    if variant not in ("", *DEMO_VARIANTS, *MODEL_VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(DEMO_VARIANTS + MODEL_VARIANTS)}")
    if variant == "shallow":  # pushed only to 0.55 rad and no presses: stays standing, leaning (0)
        right = push_moves(0.55) + [_w(2.0, *PARK_R[:2], PARK_R[2], OPEN, "aside")]
        return {"left": [_w(_duration(right), 0.10, 0.30, 0.20, OPEN, "wait")], "right": right}
    flaps = variant != "flaps_left"  # flaps_left: no flap press at all, the raised far flaps left to themselves (0.25)
    right = push_moves()
    if flaps:
        right += flap_press(FLAP_PRESS_X, FLAP_PRESS_Y, "the raised far flaps")
    left = [_w(_duration(right), *PARK_L[:2], PARK_L[2], OPEN, "wait clear of the right arm")]
    right += press_moves(PRESS_R)
    left += press_moves(PRESS_L)
    if flaps:
        right += flap_press(FLAP_FINAL_X, FLAP_PRESS_Y, "the far flaps, last", FINAL_UP_S, HOVER_PRESS)
        left += flap_press(FLAP_FINAL_X, FLAP_PRESS_YL, "the far flaps, last", FINAL_UP_S, HOVER_PRESS)
    else:
        right += [_w(2.0, *PARK_R[:2], PARK_R[2], OPEN, "aside")]
        left += [_w(2.0, *PARK_L[:2], PARK_L[2], OPEN, "aside")]
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT=shallow (a counterfactual for the README, never the submitted demo): pushed only to 0.55 rad, no
    presses: the carton stays standing and leaning (0). DEMO_VARIANT=flaps_left (also a counterfactual): the demo
    without the flap presses: racked flat but the lower layer's far flaps raised (0.25). DEMO_VARIANT=crease_set,
    crease_stiff, crease_ratchet or friction_hold: the submitted plan; the name picks the flap crease model at import
    (layout.FLAP_MODELS)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
