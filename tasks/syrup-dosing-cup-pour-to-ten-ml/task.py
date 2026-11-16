"""On the station table, real size: a 4 fl oz (118 mL) HDPE children's liquid-medicine bottle, cap off, 118 mL of
syrup in it (49 mm round body, 105 mm tall to the lip, 22 mm opening, 170 g), standing at x 30.0 cm, y -18.0 cm; a
30 mL dosing cup (36 mm square, 38 mm tall, 8 g, 2 mm walls) standing on the 100 mm pan of a pocket scale
(14 mm high, fixed) at x 30.0 cm, y 0.

Goal: 10 mL of syrup in the cup (the 2 mm printed line, 9.0 to 11.0 mL), poured with the cup held up and level, none
spilled, the cup set back on the scale and the bottle put back upright, hands off.

The syrup is a volume ledger (scoring.py): the surface of the 118 mL in the bottle follows the bottle's real pose, syrup
goes over the lip at a laminar film rate capped at 3 mL/s when the surface stands above the lip's lowest point, and
falls into the cup if that point is over its opening. The bottle and the cup are mechkit compounds (mechkit.py, next
to this file); the arms are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Trace: TASK_TRACE=<file> (one JSON line per
control step).  Calibration: DEMO_POUR_HOLD=<seconds> sets the time the bottle stays at full tilt (default 2.2).
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
import pourplan as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (tilted grips for the demo: wraps yam_arena.demo.grip_rotation)
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Pour 10 mL of syrup from the bottle into the dosing cup, holding the cup up and level while you pour. "
    "Then set the cup back on the scale and the bottle back where it stood."
)

# -- the parts -------------------------------------------------------------------------------------------------------
SCALE = Box("scale_pan", pos=(L.PAN_X, L.PAN_Y, L.TOP + L.PAN_SIZE[2] / 2), size=L.PAN_SIZE, color=L.PAN_GREY)

CUP_NAME, BOTTLE_NAME = "dosing_cup", "syrup_bottle"
# The 10 mL line: a dark strip on each of the cup's front and back faces (the hands grip the other two faces).
LINE_STRIPS = tuple(
    mk.box((0.0, s * (L.CUP_OUTER / 2 + L.LINE_T / 2), L.LINE_Z), (L.CUP_INNER, L.LINE_T, L.LINE_W), L.INK) for s in (-1, 1)
)
# The syrup's level: the one declared display rule. A thin amber plate in the cup that a slider drives, every control
# step, to the ledger's level (``syrup_target``). The scorer reads poses, not the plate.
SYRUP_PLATE = mk.slider(
    "syrup", "Z",
    [mk.box((0.0, 0.0, L.CUP_FLOOR + L.SYRUP_T / 2), (L.CUP_INNER - 0.0004, L.CUP_INNER - 0.0004, L.SYRUP_T), L.AMBER)],
    upper=L.SYRUP_RANGE, lower=0.0, mass=L.SYRUP_MASS, spring=L.SYRUP_SPRING, damping=0.3,
)
CUP = mk.assembly(
    CUP_NAME,
    origin=(L.CUP_X, L.CUP_Y, L.CUP_Z),
    mass=L.CUP_MASS,
    shapes=mk.open_box((0.0, 0.0, L.CUP_H / 2), (L.CUP_OUTER, L.CUP_OUTER, L.CUP_H), wall=L.CUP_WALL,
                       floor=L.CUP_FLOOR) + LINE_STRIPS,
    joints=[SYRUP_PLATE],
    color=L.CLEAR,
)
BOTTLE = mk.assembly(
    BOTTLE_NAME,
    origin=(L.BOTTLE_X, L.BOTTLE_Y, L.TOP),
    mass=L.BOTTLE_MASS,
    shapes=[
        mk.cylinder((0.0, 0.0, L.BODY_H / 2), L.BODY_R, L.BODY_H, "Z", L.AMBER),
        mk.cylinder((0.0, 0.0, (L.BODY_H + L.NECK_TOP) / 2), L.NECK_R, L.NECK_TOP - L.BODY_H, "Z", L.AMBER),
    ],
    color=L.AMBER,
)
OBJECTS = [SCALE, CUP, BOTTLE]

_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def syrup_target(v_cup: float) -> float:
    """The plate's travel for ``v_cup`` m^3 of syrup in the cup: its surface at V / area over the floor, never
    negative, stopped at the brim; 0 for anything that is not a finite volume."""
    try:
        v = float(v_cup)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0.0:
        return 0.0
    return min(max(v / L.CUP_AREA - L.SYRUP_T, 0.0), L.SYRUP_RANGE)


@mk.on_step
def _syrup_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    mk.set_targets(CUP, "syrup", syrup_target(_TRACKER.v_cup))


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and keeps it upright (it never tips a held bottle), and it has no
# liquids. Offline only (never in the simulator), the bottle's pose while the right hand holds it is rebuilt from the
# hand: the pad-tip point and approach axis (rigstate) and the carried point the checker follows give two directions
# that fix the bottle's rotation since the grasp (TRIAD). The scorer then runs on that, so the ledger sees the demo's
# real tilt. The cup is carried upright by the checker's own proxy, as it is held level in the simulator too.
class OfflineBottle:
    HELD_OPENING = 0.6  # a gripper closed past this (0 closed .. 1 open) on the bottle holds it

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grasp = None  # (approach axis, carried point minus TCP) at the first held step
        self.prev = None

    def state(self, bottle, arms):
        read = rigstate.read_arms(arms)
        if read is None:
            self.grasp, self.prev = None, None
            return bottle
        tcp, tips, opening = read[rigstate.ARMS.index("right")]
        centre = np.array(bottle.pos, dtype=float)  # the checker carries the box centre with the hand (any tilt)
        held = opening < self.HELD_OPENING and float(np.linalg.norm(tips - centre)) < 0.05
        if not held:
            self.grasp, self.prev = None, None
            return bottle
        ap = tips - tcp
        ap = ap / (float(np.linalg.norm(ap)) or 1.0)
        arm = centre - tcp
        if self.grasp is None:
            self.grasp = (ap, arm)
        r = _triad(self.grasp[0], self.grasp[1], ap, arm)
        origin = centre - r @ np.array([0.0, 0.0, L.NECK_TOP / 2])
        quat = P.matrix_quat(r)
        vel = np.zeros(6) if self.prev is None else np.concatenate(
            [(origin - self.prev[0]) / 0.05, _spin(self.prev[1], r) / 0.05])
        self.prev = (origin, r)
        return mk.state_from_poses(BOTTLE, (*origin, *quat), vel, source="offline")


def _frame(a, b):
    a = a / (float(np.linalg.norm(a)) or 1.0)
    c = np.cross(a, b)
    n = float(np.linalg.norm(c))
    if n < 1e-9:  # parallel: any perpendicular will do
        c = np.cross(a, np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
        n = float(np.linalg.norm(c))
    c = c / n
    return np.column_stack([a, c, np.cross(a, c)])


def _triad(a0, b0, a1, b1) -> np.ndarray:
    """The rotation that takes direction a0 to a1 and the plane (a0, b0) to the plane (a1, b1)."""
    return _frame(a1, b1) @ _frame(a0, b0).T


def _spin(r0, r1) -> np.ndarray:
    d = r1 @ r0.T
    ang = math.acos(max(-1.0, min(1.0, (float(np.trace(d)) - 1.0) / 2.0)))
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    return axis / (2.0 * math.sin(ang)) * ang


OfflineModel = OfflineBottle  # the name the other tasks use for their offline stand-in
_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(BOTTLE_NAME), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _hands_clear(arms, bottle, cup) -> bool:
    """Both grippers' finger zones 3 cm or more from both parts; False if the arms or a distance cannot be read."""
    def dist(points):
        return np.minimum(bottle.distance(points), cup.distance(points))

    try:
        return bool(rigstate.released(arms, dist, L.LET_GO))
    except Exception:  # noqa: BLE001 - fail closed: not released
        return False


def success(objects) -> bool:
    """10 mL (9.0 to 11.0) in the cup, poured with it held up and level, 0.5 mL or less spilled, the cup on the scale,
    the bottle back upright on its mark, both grippers 3 cm clear, both parts still; held 1 s. This runs first every
    control step, so it also feeds the ledger (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    bottle, cup = objects[BOTTLE_NAME], objects[CUP_NAME]
    arms = rigstate.arms()
    if no_simulator({BOTTLE_NAME: bottle}):
        bottle = _OFFLINE.state(bottle, arms)
    clear = _hands_clear(arms, bottle, cup)
    result = _TRACKER.update(bottle, cup, clear)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "resets": _TRACKER.resets, **result.details}
    try:
        row.update(bottle_origin=bottle.origin, cup_origin=cup.origin, cup_low_z=float(cup.low[2]),
                   bottle_low_z=float(bottle.low[2]), source=getattr(bottle, "source", None))
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = "unreadable part state"
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if more than 15 mL ever stood in the cup, the cup was tipped over 45 degrees, or over 10 mL was spilled. Else
    by the mL in the cup: 1.0 for 9.0 to 11.0 mL with 0.5 mL or less spilled, none poured with the cup down or tilted,
    the cup on the scale, the bottle put back and both grippers clear; 0.75 the same dose with any of that not met;
    0.5 for 8.0 to 12.0 mL (also the cap when the dose was poured with the cup not held up and level, or over 3 mL
    spilled); 0.25 for at least 4 mL; 0 below. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm ahead of them along the approach axis. Finger targets: the joint value is minus half the commanded gap (8 mm
# narrower than the part).
OPEN = -0.04695  # 94 mm
BOTTLE_GRIP = -(2 * L.BODY_R - 0.008) / 2  # 41 mm commanded across the 49 mm body
CUP_GRIP = -(L.CUP_OUTER - 0.008) / 2  # 28 mm commanded across the 36 mm cup
STAGE = 0.12  # the hands wait this far back along their approach axes
WITHDRAW = 0.16  # ... and the left hand withdraws this far, so that its fingers are 3 cm clear of the cup
WITHDRAW_R, WITHDRAW_UP = 0.10, 0.12  # the right hand: back and up (straight back runs into the arm's own base)
THETA = P.THETA_HOLD
T_UP, T_DOWN = 2.5, 2.5  # the tilt in and back (s)
POUR_HOLD = 2.2  # at full tilt (s); DEMO_POUR_HOLD overrides it for calibration (offline: 10.1 mL at 2.2 s, 2.9 mL/s)
CUP_SET_PRESS = 0.001  # the cup is lowered commanded this far below the pan's top

_YAW_R = tiltgrip.tilted(0.0, P.PITCH, 0.0)
_YAW_L = tiltgrip.tilted(-math.pi / 2, 0.0, -P.ROLL_L)


def _yaw_r(theta: float) -> float:
    return tiltgrip.tilted(0.0, P.PITCH, -theta)


def _wp(duration, pos, finger, label, yaw=0.0):
    return Waypoint(duration, tuple(float(v) for v in pos), finger, label, yaw)


def pour_hold() -> float:
    try:
        return float(os.environ.get("DEMO_POUR_HOLD", POUR_HOLD))
    except ValueError:
        return POUR_HOLD


def right_plan(hold: float | None = None) -> list[Waypoint]:
    """Right arm: grip the bottle at its body, lift it, bring it over the held cup, tilt it to 78 degrees about its
    grip point, hold, tilt it back, put it back on its mark and let go."""
    hold = pour_hold() if hold is None else hold
    a0 = P.grip_start()
    a_pour = P.pour_anchor(THETA)
    a_up = np.array([a0[0], a0[1], a_pour[2]])  # straight up from the start
    ap0 = P.right_approach(0.0)
    tcp0 = P.right_tcp(a0, 0.0)
    return [
        _wp(2.2, tcp0 - STAGE * ap0, OPEN, "beside the bottle", _YAW_R),
        _wp(1.5, tcp0, OPEN, "in to the bottle", _YAW_R),
        _wp(1.5, tcp0, BOTTLE_GRIP, "grip the bottle", _YAW_R),
        _wp(1.4, P.right_tcp(a_up, 0.0), BOTTLE_GRIP, "lift it straight up", _YAW_R),
        _wp(1.6, P.right_tcp(a_pour, 0.0), BOTTLE_GRIP, "over to the cup, upright", _YAW_R),
        _wp(T_UP, P.right_tcp(a_pour, THETA), BOTTLE_GRIP, "tilt to pour", _yaw_r(THETA)),
        _wp(hold, P.right_tcp(a_pour, THETA), BOTTLE_GRIP, "hold the pour", _yaw_r(THETA)),
        _wp(T_DOWN, P.right_tcp(a_pour, 0.0), BOTTLE_GRIP, "tilt back upright", _YAW_R),
        _wp(1.8, P.right_tcp(a0 + np.array([0.0, 0.0, 0.03]), 0.0), BOTTLE_GRIP, "back over its mark", _YAW_R),
        _wp(1.2, P.right_tcp(a0, 0.0) + np.array([0.0, 0.0, -0.0005]), BOTTLE_GRIP, "down on its mark", _YAW_R),
        _wp(1.5, P.right_tcp(a0, 0.0) + np.array([0.0, 0.0, -0.0005]), OPEN, "let go", _YAW_R),
        _wp(1.5, tcp0 - WITHDRAW_R * ap0 + np.array([0.0, 0.0, WITHDRAW_UP]), OPEN, "withdraw", _YAW_R),
    ]


def left_plan(hold: float | None = None) -> list[Waypoint]:
    """Left arm: grip the cup at its walls, lift it 45 mm and hold it level while the syrup goes in, set it down on the
    scale and let go."""
    hold = pour_hold() if hold is None else hold
    c0 = np.array([L.CUP_X, L.CUP_Y, L.CUP_Z])
    ap = P.left_approach()
    tcp0 = P.left_tcp(c0)
    up = np.array([0.0, 0.0, P.CUP_LIFT])
    # the right arm's plan, in time: the pour ends (the bottle upright again) at this many seconds
    t_right_upright = 2.2 + 1.5 + 1.5 + 1.4 + 1.6 + T_UP + hold + T_DOWN
    t_left_lifted = 2.2 + 1.5 + 1.5 + 1.2
    return [
        _wp(2.2, tcp0 - STAGE * ap, OPEN, "beside the cup", _YAW_L),
        _wp(1.5, tcp0, OPEN, "in to the cup", _YAW_L),
        _wp(1.5, tcp0, CUP_GRIP, "grip the cup", _YAW_L),
        _wp(1.2, tcp0 + up, CUP_GRIP, "lift it 45 mm", _YAW_L),
        _wp(t_right_upright - t_left_lifted, tcp0 + up, CUP_GRIP, "hold it level while the syrup goes in", _YAW_L),
        _wp(1.2, tcp0 - np.array([0.0, 0.0, CUP_SET_PRESS]), CUP_GRIP, "set it on the scale", _YAW_L),
        _wp(1.5, tcp0 - np.array([0.0, 0.0, CUP_SET_PRESS]), OPEN, "let go", _YAW_L),
        _wp(1.5, tcp0 - WITHDRAW * ap, OPEN, "withdraw", _YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": left_plan(), "right": right_plan()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
