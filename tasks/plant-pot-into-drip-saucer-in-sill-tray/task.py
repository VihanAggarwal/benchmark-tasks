"""On the station table (the window sill's top), real size: a window behind (glass in a white frame), a black sill tray
(134 x 266 mm, 12 mm raised edge) with two bays 102 mm square, formed by 3 mm stops. In the right-hand bay a loose
square drip saucer (98 mm at the rim, 12 mm high, 0.045 kg, stepped sides). In the left-hand bay another household
plant (its saucer, pot, stem and a wide canopy of leaves, 0.30 kg, one loose body) that must be left alone. Near the
right arm a sink drainer tray (130 mm square, 15 mm edge) with the watered, drained plant standing in it: a 3 in square
nursery pot (58 mm at the foot to 78 mm at the rim, 78 mm high, stepped sides, damp soil and a small succulent,
0.20 kg).

Goal: carry the pot out of the drainer, over the tray's edge and the saucer's rim, set it down in the saucer's well (72
mm square, the foot 58 mm: 7 mm each side), let go and take the hands away, leaving the saucer in its bay and the
neighbouring plant where it was.

The pot, the saucer, the neighbour, the tray, the drainer and the window are mechkit compounds (mechkit.py, next to
this file). No software rules: every rule is the scorer's reading of poses. The arms are read through rigstate.py
(the let-go check).

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
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import NEIGHBOUR, POT, SAUCER  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "The watered houseplant has drained in the sink drainer. Carry it to the window sill and set its pot down in its "
    "saucer in the tray, then let go. Leave the saucer in its bay and the other plant where it is."
)


# -- the parts --------------------------------------------------------------------------------------------------------
def ring(zc, h, outer, inner, color=None):
    """Four boxes round a square hole: ``outer`` and ``inner`` widths, ``h`` tall, centred at height ``zc``."""
    w = (outer - inner) / 2
    d = (outer - w) / 2
    return [mk.box((s * d, 0.0, zc), (w, outer, h), color) for s in (-1, 1)] + \
           [mk.box((0.0, s * d, zc), (inner, w, h), color) for s in (-1, 1)]


def saucer_shapes(z0=0.0):
    """The saucer: a 92 mm floor, a lower wall (72 mm well) to 8 mm, a flared upper wall (84 mm mouth, 98 mm over all)
    to 12 mm. Base frame at the centre of its underside."""
    c = L.SAU_TERRACOTTA
    return [mk.box((0.0, 0.0, z0 + L.SAU_FLOOR / 2), (L.SAU_BASE, L.SAU_BASE, L.SAU_FLOOR), c)] + \
        ring(z0 + (L.SAU_FLOOR + L.SAU_STEP) / 2, L.SAU_STEP - L.SAU_FLOOR, L.SAU_BASE, L.SAU_WELL, c) + \
        ring(z0 + (L.SAU_STEP + L.SAU_H) / 2, L.SAU_H - L.SAU_STEP, L.SAU_RIM, L.SAU_MOUTH, c)


def pot_shapes(z0=0.0, plant=True):
    """The pot: three stepped layers (58, 65, 72 mm) standing in for the taper, a 6 mm rim ring 78 mm over all, the
    soil 2 mm over the top layer and, for the watered plant, the succulent on it. Base frame at the centre of the
    foot's underside (``z0`` lifts it, for the neighbour's pot standing in its saucer)."""
    out = [mk.box((0.0, 0.0, z0 + (za + zb) / 2), (w, w, zb - za), L.POT_BLACK) for za, zb, w in L.POT_LAYERS]
    out += ring(z0 + (L.POT_RIM_Z0 + L.POT_H) / 2, L.POT_H - L.POT_RIM_Z0, L.POT_RIM, L.POT_LAYERS[-1][2], L.POT_BLACK)
    out.append(mk.box((0.0, 0.0, z0 + (L.POT_RIM_Z0 + L.POT_SOIL_TOP) / 2),
                      (L.POT_LAYERS[-1][2] - 0.002, L.POT_LAYERS[-1][2] - 0.002, L.POT_SOIL_TOP - L.POT_RIM_Z0),
                      L.SOIL_BROWN))
    if plant:
        out.append(mk.cylinder((0.0, 0.0, z0 + L.POT_SOIL_TOP + L.PLANT_H / 2), L.PLANT_R, L.PLANT_H, "Z",
                               L.PLANT_GREEN))
    return out


def neighbour_shapes():
    """The other plant as one body: saucer, pot standing in it, a stem and a wide canopy of leaves."""
    stem_z0 = L.SAU_FLOOR + L.POT_SOIL_TOP
    stem_z1 = L.NB_CANOPY_Z - L.NB_CANOPY_T / 2
    return saucer_shapes() + pot_shapes(L.SAU_FLOOR, plant=False) + [
        mk.box((0.0, 0.0, (stem_z0 + stem_z1) / 2), (L.NB_STEM, L.NB_STEM, stem_z1 - stem_z0), L.LEAF_GREEN),
        mk.cylinder((0.0, 0.0, L.NB_CANOPY_Z), L.NB_CANOPY_R, L.NB_CANOPY_T, "Z", L.LEAF_GREEN),
    ]


def tray_shapes():
    """The sill tray (base frame at the centre of its underside): floor, raised edge, and the stops all round each bay."""
    ox, oy = L.TRAY_OUTER
    out = list(mk.open_box((0.0, 0.0, L.TRAY_H / 2), (ox, oy, L.TRAY_H), L.TRAY_WALL, L.TRAY_FLOOR, L.TRAY_BLACK))
    for by in (L.BAY_B[1] - L.TRAY_Y, L.BAY_A[1] - L.TRAY_Y):
        zc = L.TRAY_FLOOR + L.STOP_H / 2
        for s in (-1, 1):
            out.append(mk.box((s * (L.BAY / 2 + L.STOP_T / 2), by, zc), (L.STOP_T, L.BAY + 2 * L.STOP_T, L.STOP_H),
                              L.TRAY_BLACK))
            out.append(mk.box((0.0, by + s * (L.BAY / 2 + L.STOP_T / 2), zc), (L.BAY, L.STOP_T, L.STOP_H),
                              L.TRAY_BLACK))
    return out


def drainer_shapes():
    return list(mk.open_box((0.0, 0.0, L.DRAIN_H / 2), (L.DRAIN_OUTER, L.DRAIN_OUTER, L.DRAIN_H), L.DRAIN_WALL,
                            L.DRAIN_FLOOR, L.DRAIN_GREY))


def window_shapes():
    """Base frame at the sill's top, at the glass's centre line: a bottom and a top rail, two jambs and the pane."""
    gh = L.GLASS_Z1 - L.GLASS_Z0
    wide = L.GLASS_W + 2 * L.FRAME_T
    return [
        mk.box((0.0, 0.0, L.GLASS_Z0 / 2), (L.FRAME_DEPTH, wide, L.GLASS_Z0), L.FRAME_WHITE),
        mk.box((0.0, 0.0, L.GLASS_Z1 + L.FRAME_T / 2), (L.FRAME_DEPTH, wide, L.FRAME_T), L.FRAME_WHITE),
        mk.box((0.0, (L.GLASS_W + L.FRAME_T) / 2, L.GLASS_Z0 + gh / 2), (L.FRAME_DEPTH, L.FRAME_T, gh), L.FRAME_WHITE),
        mk.box((0.0, -(L.GLASS_W + L.FRAME_T) / 2, L.GLASS_Z0 + gh / 2), (L.FRAME_DEPTH, L.FRAME_T, gh), L.FRAME_WHITE),
        mk.box((0.0, 0.0, L.GLASS_Z0 + gh / 2), (L.GLASS_T, L.GLASS_W, gh), L.GLASS_BLUE),
    ]


WINDOW = mk.assembly("window", origin=(L.WIN_X, 0.0, L.TOP), shapes=window_shapes(), color=L.FRAME_WHITE)
TRAY = mk.assembly("tray", origin=(L.TRAY_X, L.TRAY_Y, L.TOP), shapes=tray_shapes(), color=L.TRAY_BLACK,
                   friction=L.FRICTION, tight_fit=True)
DRAINER = mk.assembly("drainer", origin=(L.POT_X0, L.POT_Y0, L.TOP), shapes=drainer_shapes(), color=L.DRAIN_GREY,
                      friction=L.FRICTION, tight_fit=True)
SAUCER_PART = mk.assembly(SAUCER, origin=L.SAU_ORIGIN, mass=L.SAU_MASS, shapes=saucer_shapes(),
                          color=L.SAU_TERRACOTTA, friction=L.FRICTION, tight_fit=True)
POT_PART = mk.assembly(POT, origin=L.POT_ORIGIN, mass=L.POT_MASS, shapes=pot_shapes(), color=L.POT_BLACK,
                       friction=L.FRICTION, tight_fit=True)
NEIGHBOUR_PART = mk.assembly(NEIGHBOUR, origin=L.NB_ORIGIN, mass=L.NB_MASS, shapes=neighbour_shapes(),
                             color=L.LEAF_GREEN, friction=L.FRICTION, tight_fit=True)
OBJECTS = [WINDOW, TRAY, DRAINER, SAUCER_PART, POT_PART, NEIGHBOUR_PART]
SPECS = {POT: POT_PART, SAUCER: SAUCER_PART, NEIGHBOUR: NEIGHBOUR_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped box rigidly and drops it onto the bounding box below; it knows nothing of the
# saucer's well. Offline only (never in the simulator), this model gives the scorer the pot: a closed gripper whose TCP
# is within 12 mm of the pot's axis and 20 to 52 mm above its foot takes it up and carries it rigidly; opened, the
# pot drops straight down onto what is under its foot: the saucer's well floor if the foot fits the well (7 mm each
# side), the saucer's rim if it only overlaps the saucer, the tray's floor inside its edge, the drainer's floor, or
# the table. The saucer and the neighbour never move offline.
HOLD_OPEN = 0.80  # opening (0 closed .. 1 open) at or below this: closed on the pot
LET_OPEN = 0.85  # above this: let go


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pot = np.array(L.POT_ORIGIN, dtype=float)
        self.held = None
        self.off = np.zeros(3)

    def support(self, x: float, y: float) -> float:
        """Height of the surface under a pot foot centred at (x, y)."""
        so = np.array(L.SAU_ORIGIN)
        slack = L.SAU_WELL / 2 - L.POT_BASE / 2
        dx, dy = abs(x - so[0]), abs(y - so[1])
        if max(dx, dy) <= slack:
            return L.POT_SEATED_Z
        if max(dx, dy) <= L.SAU_RIM / 2 + L.POT_BASE / 2:
            return so[2] + L.SAU_H  # standing on the rim
        if abs(x - L.TRAY_X) <= L.TRAY_INNER[0] / 2 and abs(y - L.TRAY_Y) <= L.TRAY_INNER[1] / 2:
            return L.TOP + L.TRAY_FLOOR
        if abs(x - L.POT_X0) <= (L.DRAIN_OUTER / 2 - L.DRAIN_WALL) and abs(y - L.POT_Y0) <= (L.DRAIN_OUTER / 2 - L.DRAIN_WALL):
            return L.TOP + L.DRAIN_FLOOR
        return L.TOP

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        if self.held is None:
            for side, (tcp, _tips, opening) in zip(rigstate.ARMS, read):
                rel = tcp - self.pot
                if opening <= HOLD_OPEN and math.hypot(rel[0], rel[1]) <= 0.012 and 0.020 <= rel[2] <= 0.052:
                    self.held, self.off = side, self.pot - tcp
                    break
            return
        tcp, _tips, opening = read[rigstate.ARMS.index(self.held)]
        if opening > LET_OPEN:
            self.held = None
            self.pot[2] = self.support(self.pot[0], self.pot[1])
        else:
            self.pot = tcp + self.off

    def states(self) -> dict:
        q = (0.0, 0.0, 0.0, 1.0)
        return {
            POT: mk.state_from_poses(POT_PART, (*self.pot, *q), source="offline"),
            SAUCER: mk.state_from_poses(SAUCER_PART, (*L.SAU_ORIGIN, *q), source="offline"),
            NEIGHBOUR: mk.state_from_poses(NEIGHBOUR_PART, (*L.NB_ORIGIN, *q), source="offline"),
        }


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(POT), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, foot dz max mm, pot off-centre mm, nb move mm]


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
    """The pot seated in the saucer, the saucer in its bay, the neighbour never disturbed, both grippers 30 mm clear and
    all three parts still for 1 s; held 1 s. This runs first every control step, so it also feeds the tracker (and,
    without a simulator, the offline stand-in)."""
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
    row["source"] = getattr(parts.get(POT), "source", None)
    d = result.details
    if d.get("valid_state") and _TRACKER.steps % 2 == 0:
        _HIST.append([_TRACKER.steps, d["foot_dz_max_mm"], d["pot_centre_off_mm"], d["nb_move_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the pot tipped more than 30 deg, a part left the sill, the neighbour toppled, or unless every part is let go
    and still; 1.0 the pot seated in the saucer, the saucer in its bay, the neighbour never moved 3 mm or tilted 3 deg;
    0.5 the pot seated otherwise; 0.25 the pot upright inside the tray's outline, not seated; 0 otherwise. The score of
    the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
OPEN = -0.04695  # 94 mm
POT_GRIP = -(L.GRIP_W - 0.008) / 2  # 57 mm commanded across the pot's 65 mm second layer
YAW_R = math.pi / 2  # the right arm closes along x (front to back), which keeps its housing off the neighbour's canopy

GRIP_H = L.GRIP_Z0 + 0.0005 + L.PAD_TIP  # TCP height over the pot's foot: the pads (22 mm) fill the second layer
GRIP_TCP = L.POT_ORIGIN[2] + GRIP_H
LIFT = 0.032  # commanded: the foot 35 mm over the table, 20 mm over the saucer's rim (15 mm), before any trail
CARRY_TCP = GRIP_TCP + LIFT
PRESS = 0.002  # lowered 2 mm past the well floor, so the pot rests its full weight there before the fingers open
SET_TCP = L.POT_SEATED_Z + GRIP_H - PRESS
APPROACH = 0.030  # each final descent starts this far above its target and takes 1.5 to 2 s
HOVER_R = L.TOP + 0.115  # pad tips 102 mm over the table: over the succulent (100 mm over the drainer's floor)


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, z), finger, label, YAW_R)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def right_plan():
    """The right arm: over the pot in the drainer, down round its second layer, grip, lift 32 mm, carry over the right
    bay at that height, lower into the well, let go, straight up, away over the drainer."""
    px, py = L.POT_X0, L.POT_Y0
    bx, by = L.BAY_B
    return [
        _w(2.5, px, py, HOVER_R, OPEN, "over the pot"),
        _w(1.0, px, py, GRIP_TCP + APPROACH, OPEN, "down to just above the grip"),
        _w(1.5, px, py, GRIP_TCP, OPEN, "slowly down round the pot"),
        _w(0.5, px, py, GRIP_TCP, OPEN, "settle"),
        _w(1.5, px, py, GRIP_TCP, POT_GRIP, "grip the pot across its second layer"),
        _w(0.5, px, py, GRIP_TCP, POT_GRIP, "settle the grip"),
        _w(1.5, px, py, CARRY_TCP, POT_GRIP, "lift it out of the drainer"),
        _w(3.0, bx, by, CARRY_TCP, POT_GRIP, "carry it over the saucer"),
        _w(0.5, bx, by, CARRY_TCP, POT_GRIP, "settle"),
        _w(1.0, bx, by, SET_TCP + APPROACH, POT_GRIP, "down to just over the saucer's rim"),
        _w(2.0, bx, by, SET_TCP, POT_GRIP, "slowly down into the well"),
        _w(0.5, bx, by, SET_TCP, POT_GRIP, "settle"),
        _w(1.5, bx, by, SET_TCP, OPEN, "let go"),
        _w(1.5, bx, by, HOVER_R, OPEN, "straight up"),
        _w(2.0, px, py, HOVER_R, OPEN, "away over the drainer"),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check). The left arm has no plan: it stays at its
    ready pose with the gripper open, out of the way."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"right": right_plan()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
