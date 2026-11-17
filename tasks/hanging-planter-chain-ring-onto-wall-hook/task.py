"""On the station table, real size: a 4 in galvanized pail planter (100 mm across the rim, 80 mm tall, 0.25 kg with 170 g
of sand) stands at x 250 mm, y 100 mm. Its 4 mm wire bail is hinged on two ears at the rim (axis along y) and a 2 in
zinc ring (40 mm hole, 5 mm wire) is linked through the bail's apex (axis along x), so the pot swings from the ring on
two hinges. Beyond it, 430 mm in front of the arm bases, a 20 mm plywood wall panel stands on the table with a steel
plant bracket on its face: a plate, an 8 mm bar projecting 150 mm toward the robot at y 200 mm, z 190 mm, and a lip
standing 10 mm above the bar's top at its free end.

Goal: the ring on the bar behind the lip with the planter hanging from it (the pot's bottom 27.5 mm above the table
when plumb), both grippers 30 mm clear, and the pot and ring still for 1 s. The ring has to be lifted
over the lip (its hole is 40 mm; the bar and the lip's top stand within 10 to 12 mm of its walls), moved in along the
bar, and let down onto it. Lifted by the ring the pot swings.

The planter, the wall panel and the bracket are mechkit compounds (mechkit.py, next to this file). No software rule
touches the planter: the ring and the bail are plain hinge joints with a light spring toward upright (the bail's own
stiffness). The arms are read through rigstate.py (the let-go check).

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
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import PLANTER  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Hang the planter on the wall hook by its ring so it hangs clear of the table, and let go of it."


# -- the parts --------------------------------------------------------------------------------------------------------
def pot_shapes():
    return [mk.cylinder((0.0, 0.0, L.POT_BODY_H / 2), L.POT_R, L.POT_BODY_H, "Z", L.GALV),
            mk.cylinder((0.0, 0.0, L.POT_BODY_H + L.RIM_H / 2), L.RIM_R, L.RIM_H, "Z", L.GALV)]


def bail_shapes():
    """Two side wires up from the ears and the top bar, 4 mm wire, in the pot's frame at rest."""
    w = L.BAIL_WIRE
    mid = (L.EAR_Z + L.APEX_Z) / 2
    out = [mk.box((0.0, s * L.BAIL_Y, mid), (w, w, L.BAIL_RISE + w), L.ZINC) for s in (-1, 1)]
    out.append(mk.box((0.0, 0.0, L.APEX_Z), (w, 2 * L.BAIL_Y + w, w), L.ZINC))
    return out


def ring_shapes():
    """The ring as an octagon of eight 5 mm wire segments in the plane x = 0 (hole axis along x), wire centre line 22.5
    mm from its centre, hole 40 mm across flats; its centre 22.5 mm above the bail's apex."""
    side = 2.0 * (L.RING_APOTHEM + L.RING_WIRE / 2) * math.tan(math.pi / 8)  # a little over a side: the corners overlap
    out = []
    for k in range(8):
        phi = k * math.pi / 4
        c = (0.0, L.RING_APOTHEM * math.sin(phi), L.RING_Z0 + L.RING_APOTHEM * math.cos(phi))
        out.append(mk.box(c, (L.RING_WIRE, side, L.RING_WIRE), L.ZINC, ypr=(0.0, 0.0, -phi)))
    return out


BAIL = mk.hinge("bail", "Y", (0.0, 0.0, L.EAR_Z), bail_shapes(), lower=-L.JOINT_LIMIT, upper=L.JOINT_LIMIT,
                mass=L.BAIL_MASS, spring=L.BAIL_SPRING, damping=L.BAIL_DAMP)
RING = mk.hinge(S.RING, "X", (0.0, 0.0, L.APEX_Z), ring_shapes(), lower=-L.JOINT_LIMIT, upper=L.JOINT_LIMIT,
                mass=L.RING_MASS, spring=L.RING_SPRING, damping=L.RING_DAMP, parent="bail")


def wall_shapes():
    """In the table-top frame: the panel, the bracket's plate, its round bar along x and the lip at the free end."""
    return [
        mk.box(((L.WALL_X0 + L.WALL_X1) / 2, (L.WALL_Y0 + L.WALL_Y1) / 2, L.WALL_H / 2),
               (L.WALL_X1 - L.WALL_X0, L.WALL_Y1 - L.WALL_Y0, L.WALL_H), L.PLYWOOD),
        mk.box((L.WALL_X0 - L.PLATE_T / 2, L.HOOK_Y, L.HOOK_Z), (L.PLATE_T, L.PLATE_W, L.PLATE_H), L.STEEL),
        mk.cylinder(((L.BAR_X0 + L.BAR_X1) / 2, L.HOOK_Y, L.HOOK_Z), L.BAR_R, L.BAR_X1 - L.BAR_X0, "X", L.STEEL),
        mk.cylinder((L.LIP_X, L.HOOK_Y, (L.LIP_Z0 + L.LIP_Z1) / 2), L.LIP_R, L.LIP_Z1 - L.LIP_Z0, "Z", L.STEEL),
    ]


PLANTER_PART = mk.assembly(PLANTER, origin=(L.POT_X, L.POT_Y, L.TOP), mass=L.POT_MASS, shapes=pot_shapes(),
                           joints=[BAIL, RING], color=L.GALV, friction=L.FRICTION, solver_iterations=32)
WALL = mk.assembly("wall", origin=(0.0, 0.0, L.TOP), shapes=wall_shapes(), color=L.PLYWOOD)
OBJECTS = [WALL, PLANTER_PART]
SPECS = {PLANTER: PLANTER_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it knows nothing of the bracket. Offline
# only (never in the simulator), this model gives the scorer: the ring follows the left gripper once it has closed on it
# (fingers closed to the ring's width, the TCP within 20 mm of its centre sideways and 12 mm of grasp height), the pot
# hangs plumb below it; let go with the bar through the ring's hole behind the lip the ring drops onto the bar (centre
# 16 mm below its axis) and the planter hangs there; let go anywhere else it falls straight down and stands on the
# table. A hand holding the ring is not a hand holding the planter up: only the ring is read.
GRASP_UP = 0.005  # the TCP 5 mm above the ring's centre: the pad tips 8 mm below it, over the bail's top bar
GRIP_OPEN_MAX = 0.55  # the fingers count as closed on the ring below this opening (0 closed .. 1 open)


def _over_bar(c) -> bool:
    """A ring centre at which the bar's axis passes through its hole behind the lip, ring plane square to the bar."""
    return bool(L.BAR_BEHIND_LIP <= c[0] <= L.BAR_X1 - L.BAR_END_MARGIN and abs(c[1] - L.HOOK_Y) <= L.THROUGH_R - L.BAR_R
                and L.HOOK_Z + L.BAR_R - L.RING_IN - 1e-6 <= c[2] <= L.HOOK_Z - L.BAR_R + L.RING_IN + 1e-6)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ring = np.array([L.POT_X, L.POT_Y, L.RING_Z0])  # the ring's centre (table-top frame)
        self.grip = None  # offset of the ring from the TCP while held

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        tcp, _, opening = read[0]
        tcp = tcp - np.array([0.0, 0.0, L.TOP])
        closed = opening < GRIP_OPEN_MAX
        if self.grip is None:
            near = (np.linalg.norm(tcp[:2] - self.ring[:2]) <= 0.020
                    and abs(tcp[2] - (self.ring[2] + GRASP_UP)) <= 0.012)
            if closed and near:
                self.grip = self.ring - tcp
        elif not closed:
            self.grip = None
            self.settle()
        if self.grip is not None:
            self.ring = tcp + self.grip
            if _over_bar(np.array([self.ring[0], self.ring[1], L.RING_HUNG_Z])) and self.ring[2] < L.RING_HUNG_Z:
                self.ring[2] = L.RING_HUNG_Z  # the bar stops the ring: the hand slips on it, it does not go lower

    def settle(self) -> None:
        c = self.ring.copy()
        if _over_bar(c):
            c[2] = L.RING_HUNG_Z  # the ring's upper wire comes down on the bar
        else:
            c[2] = L.RING_Z0  # falls: the planter stands on the table under where the ring was
        self.ring = c

    def states(self) -> dict:
        pot = self.ring - np.array([0.0, 0.0, L.RING_Z0])
        pose = (pot[0], pot[1], L.TOP + pot[2], 0.0, 0.0, 0.0, 1.0)
        return {PLANTER: mk.state_from_poses(PLANTER_PART, pose, source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(PLANTER), "source", "sim") != "sim"


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
    """With TASK_TRACE=<file>, append one JSON line per control step: the planter's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The ring on the bar behind the lip, the planter hanging from it with the pot 20 mm or more above the table and
    plumb, both grippers 30 mm clear and the pot and ring still for 1 s, no latched fall; held 1 s. This runs first
    every control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {PLANTER: objects.get(PLANTER)}
    arms = rigstate.arms()
    if no_simulator(parts):
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    s = parts.get(PLANTER)
    try:
        row["joints"] = {k: round(float(v), 4) for k, v in s.joints.items()}
        row["joint_vel"] = {k: round(float(v), 4) for k, v in s.joint_vel.items()}
    except (AttributeError, TypeError, ValueError):
        row["joints"] = None
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(s, "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the pot ever fell off the table, or unless both grippers are 30 mm clear for the last 1 s, or the ring is
    not on the bar behind the lip; 1.0 hanging from the bar, the pot 20 mm or more above the table, plumb within 10
    degrees, the pot and ring still for 1 s; 0.5 the same but leaning or still swinging; 0.25 on the bar with the pot
    touching the table or the wall. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
RING_OD = 2 * (L.RING_APOTHEM + L.RING_WIRE / 2)  # 50 mm across the flats
GRIP = -(RING_OD - 0.008) / 2  # 42 mm commanded across the ring's 50 mm
YAW = 0.0  # fingers close along y: on the ring's two vertical flats

GRASP_TCP = L.RING_Z0 + GRASP_UP  # 151.5 mm
HOVER = 0.185
TRANSIT_TCP = L.RING_THREAD_Z + GRASP_UP  # 199 mm: the ring's centre 4 mm over the bar's axis, the lip inside the hole
X_THREAD0 = 0.250  # the ring's x when it reaches the hook's line: its plane 21 mm short of the lip's centre
X_HANG = 0.310  # the ring's x when hung: the bar crosses it 23 mm behind the lip's rear edge
PRESS = 0.004  # lowered this far past where the ring comes to rest on the bar (the arm trails by more under load)
HUNG_TCP = L.RING_HUNG_Z + GRASP_UP - PRESS  # 175 mm
X_RETREAT = 0.170  # back along x, out of the ring's way, the finger zones 30 mm clear of the planter
X_CLEAR = 0.225  # the way in from the ready pose stays this far short of the bracket's lip (x 270 mm): the open pads (94 mm
# apart, the outer one 47 mm from the TCP) would otherwise land on the bar (z 194 mm at its top) while the hand
# crosses y 200 mm lower than 217 mm: the first GPU run stopped with the hand at x 287, y 157 mm for that reason
Y_ENTRY = 0.305  # the left arm's own column (the ready pose's y): nothing stands there


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, YAW)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


# Wrong ways (counterfactuals): the ring let down in front of the lip (the planter falls to the table), and never let go.
VARIANTS = ("front", "hold")


def left_plan(variant: str = "") -> list:
    """The left arm: round in front of the lip to the ring, down round the ring, grip its two sides, lift the planter 47 mm, carry it 100 mm toward the hook's
    line, move the ring in along the bar over the lip, let it down onto the bar, open, back off along x."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    x0, y0 = L.POT_X, L.POT_Y
    x_end = X_THREAD0 if variant == "front" else X_HANG
    plan = [
        _w(1.5, X_CLEAR, Y_ENTRY, HOVER, OPEN, "down beside the bracket's end, short of the bar and the lip"),
        _w(1.5, X_CLEAR, y0, HOVER, OPEN, "across, in front of the lip, the pads 47 mm either side"),
        _w(1.0, x0, y0, HOVER, OPEN, "over the ring"),
        _w(1.0, x0, y0, GRASP_TCP + 0.020, OPEN, "down to just above the grip"),
        _w(1.5, x0, y0, GRASP_TCP, OPEN, "slowly down beside the ring's flats"),
        _w(0.5, x0, y0, GRASP_TCP, OPEN, "settle"),
        _w(1.5, x0, y0, GRASP_TCP, GRIP, "grip the ring"),
        _w(0.5, x0, y0, GRASP_TCP, GRIP, "settle the grip"),
        _w(1.5, x0, y0, TRANSIT_TCP, GRIP, "lift the planter off the table"),
        _w(2.5, X_THREAD0, L.HOOK_Y, TRANSIT_TCP, GRIP, "carry it to the hook's line, in front of the lip"),
        _w(1.0, X_THREAD0, L.HOOK_Y, TRANSIT_TCP, GRIP, "let the swing die"),
    ]
    if variant != "front":
        plan += [
            _w(2.0, x_end, L.HOOK_Y, TRANSIT_TCP, GRIP, "move the ring in along the bar, over the lip"),
            _w(1.0, x_end, L.HOOK_Y, TRANSIT_TCP, GRIP, "let the swing die"),
        ]
    plan += [
        _w(1.5, x_end, L.HOOK_Y, HUNG_TCP, GRIP, "let the ring down onto the bar"),
        _w(1.0, x_end, L.HOOK_Y, HUNG_TCP, GRIP, "settle: the bar takes the weight"),
    ]
    if variant == "hold":
        plan.append(_w(3.0, x_end, L.HOOK_Y, HUNG_TCP, GRIP, "keep holding the ring"))
        return plan
    plan += [
        _w(1.5, x_end, L.HOOK_Y, HUNG_TCP, OPEN, "let go"),
        _w(2.0, X_RETREAT, L.HOOK_Y, HUNG_TCP, OPEN, "back off along x"),
        _w(0.5, X_RETREAT, L.HOOK_Y, HUNG_TCP, OPEN, "hold"),
    ]
    return plan


def demo():
    """The left arm alone does the job; the right arm holds the ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold
    the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      front  the ring let down 60 mm short of the lip and let go: the planter falls to the table (0)
      hold   everything as in the demo but the fingers never open: a held planter earns 0"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": left_plan(os.environ.get("DEMO_VARIANT", ""))}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
