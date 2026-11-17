"""On the station table, real size: a stack of three knocked-down 12 x 8 x 6 in cartons (508 x 152 x 6.4 mm each) on a
fixed brown board at x 270 mm, centred on y 0. On top of the stack lies an empty carton of the same size, flaps cut
off, on one long side with its open ends along x: a 305 (y) x 152 (x) x 203 mm tall four-panel sleeve in 3.2 mm
single-wall C-flute. Its corners are creases (rotating joints about axes along x); the fourth corner is the
manufacturer's glue joint, a 38 mm tab creased on the end panel's edge and glued (a mechkit bond, a FixedJoint that
never breaks) to the underneath panel, so the sleeve is a closed four-bar.

Goal: the carton collapsed flat (every crease within 15 degrees of shut or open, 12 mm thick at most) lying on the
stack, the stack still as found, hands off. Racking the sleeve toward +y by pushing its -y end panel turns the four
creases together; past about 45 degrees its own weight finishes the job, and a press along the top seats the layers.

The carton's panels and the stack's flats are mechkit compounds (mechkit.py, next to this file). The arms are read
through rigstate.py (the let-go check). There is no declared software rule.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactual (never the submitted demo):
DEMO_VARIANT=shallow.  Trace: TASK_TRACE=<file> (one JSON line per control step).
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


# -- the parts --------------------------------------------------------------------------------------------------------
def _panel(body):
    centre, size, _ = L.PANELS[body]
    return mk.box(centre, size, L.CARTON_BROWN)


def _crease(body):
    return mk.hinge(body, "X", L.ANCHOR[body], [_panel(body)], upper=L.CREASE_LIMIT, lower=-L.CREASE_LIMIT,
                    mass=L.PANELS[body][2], damping=L.CREASE_DAMPING, friction=L.CREASE_FRICTION,
                    parent=L.PARENT[body])


BOARD = mk.assembly("board", origin=(L.PILE_X, L.PILE_Y, L.TOP),
                    shapes=[mk.box((0.0, 0.0, L.BOARD_T / 2), (L.BOARD_X, L.BOARD_Y, L.BOARD_T), L.BOARD_BROWN)],
                    color=L.BOARD_BROWN, friction=L.FLAT_FRICTION, tight_fit=True)
FLAT_PARTS = [mk.assembly(FLATS[i], origin=(L.PILE_X, L.PILE_Y, L.TOP + L.flat_z(i)), mass=L.FLAT_MASS,
                          shapes=[mk.box((0.0, 0.0, 0.0), (L.FLAT_X, L.FLAT_Y, L.FLAT_T), L.PILE_BROWN)],
                          color=L.PILE_BROWN, friction=L.FLAT_FRICTION, tight_fit=True) for i in range(L.FLAT_N)]
CARTON_PART = mk.assembly(CARTON, origin=(L.CARTON_X, L.CARTON_Y, L.TOP + L.CARTON_ORIGIN_Z), mass=L.PANELS["base"][2],
                          shapes=[_panel("base")], joints=[_crease(b) for b in L.CREASES], color=L.CARTON_BROWN,
                          friction=L.CARTON_FRICTION, tight_fit=True)
SEAM = mk.bond("seam_glue", (CARTON_PART, "tab"), CARTON_PART)  # the glue joint: the tab glued to A
OBJECTS = [BOARD, *FLAT_PARTS, CARTON_PART]
SPECS = {CARTON: CARTON_PART, **{FLATS[i]: FLAT_PARTS[i] for i in range(L.FLAT_N)}}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it knows nothing of creases. Offline
# only (never in the simulator), this model gives the scorer the carton racked toward +y by one angle PHI (the rack
# of the four-bar sleeve; see layout.py): a closed gripper whose pad face pushes the -y end panel's outer face at a
# height ``hz`` over the stack's top, moved ``dy`` past the panel's foot, leans it to atan(dy / hz); once the lean
# passes GRAVITY_PHI the carton's own weight carries it on at 1.2 rad/s to flat. The stack never moves. PHI never
# decreases.
PAD_FACE = 0.0059  # a closed pad block's face ahead of the TCP along the closing direction (GPU: 5.9 mm, shim task)
D_FACE_Y = L.A_YMIN - L.CT / 2  # the -y end panel's outer face at its foot
GRAVITY_PHI = 0.80  # rad (46 degrees): past this the weight finishes the rack (offline assumption; see README)
FALL_RATE = 1.2  # rad/s
HZ_MIN, HZ_MAX = 0.04, 0.19  # the pad block touches the end panel between these heights over the stack's top


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phi = 0.0

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is not None:
            for tcp, _tips, opening in read:
                hz = float(tcp[2]) - L.TOP - L.PT
                if opening > 0.3 or not (HZ_MIN <= hz <= HZ_MAX):
                    continue
                if abs(float(tcp[0]) - L.CARTON_X) > L.LX / 2 + 0.02:
                    continue
                dy = float(tcp[1]) + PAD_FACE - D_FACE_Y
                if dy <= 0.0 or float(tcp[1]) > L.A_YMAX:
                    continue
                self.phi = max(self.phi, min(math.pi / 2, math.atan2(dy, hz)))
        if GRAVITY_PHI <= self.phi < math.pi / 2:
            self.phi = min(math.pi / 2, self.phi + FALL_RATE / L.CONTROL_HZ)

    def states(self) -> dict:
        out = {CARTON: mk.state_from_poses(CARTON_PART, (L.CARTON_X, L.CARTON_Y, L.TOP + L.CARTON_ORIGIN_Z,
                                                         0.0, 0.0, 0.0, 1.0), joints=L.joints_for(self.phi),
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


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


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
        row["creases_rad"] = {k: round(float(parts[CARTON].joints[k]), 4) for k in L.CREASES}
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
    row["source"] = getattr(parts.get(CARTON), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any flat of the stack was moved over 40 mm or turned over 30 degrees, or unless every part is let go and
    still; 1.0 the carton flat on the stack with the stack kept; 0.75 flat on it but offset or leaning; 0.5 flat
    anywhere else; 0.25 not flat but no more than 110 mm tall; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Gripper closing axis along y (yaw 0). Finger targets: the joint value is minus half the commanded gap.
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
PRESS_R = (-0.14, -0.02)  # the right arm's two press points along y (its own side, up to the midline)
PRESS_L = (0.12, 0.24)  # the left arm's
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


def press_moves(ys):
    """Closed block pressed 1 mm into the flat carton's top at each y, then lifted."""
    plan = []
    for y in ys:
        plan += [
            _w(1.5, XC, y, HOVER_PRESS, CLOSED, f"over y {y * 1000:.0f} mm"),
            _w(1.2, XC, y, PRESS_Z, CLOSED, f"press down at y {y * 1000:.0f} mm"),
            _w(0.5, XC, y, PRESS_Z, CLOSED, "hold the press"),
            _w(1.0, XC, y, HOVER_PRESS, CLOSED, "up"),
        ]
    return plan


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the right arm pushes the end panel over (the weight finishes it), lifts clear and presses along the
    top on its side; then the left arm presses along the top on its side; both park clear."""
    if variant not in ("", "shallow"):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of shallow")
    if variant == "shallow":  # pushed only to 0.55 rad and no presses: stays standing, leaning (0)
        right = push_moves(0.55) + [_w(2.0, *PARK_R[:2], PARK_R[2], OPEN, "aside")]
        return {"left": [_w(_duration(right), 0.10, 0.30, 0.20, OPEN, "wait")], "right": right}
    right = push_moves() + press_moves(PRESS_R) + [_w(2.0, *PARK_R[:2], PARK_R[2], OPEN, "aside")]
    waited = _duration(right) - 2.0
    left = [_w(waited, *PARK_L[:2], PARK_L[2], OPEN, "wait clear of the right arm")]
    left += press_moves(PRESS_L) + [_w(2.0, *PARK_L[:2], PARK_L[2], OPEN, "aside")]
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT=shallow (a counterfactual for the README, never the submitted demo): pushed only to 0.55 rad, no
    presses: the carton stays standing and leaning (0)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
