"""On the station table, real size: a dog CPR manikin (Simulaids CasPeR, 711 x 584 x 254 mm, 6.8 kg) lies loose on its
right side, head toward the arms, wearing a red 3/4 in nylon collar (size M) closed with a side-release buckle (black
male, dark grey female) on top of its neck. The collar's loop (381 mm) is 54 mm longer than the neck's round (327 mm)
and 34 mm shorter than the head's round behind the ears, so the buckled collar cannot come off over the head.

Goal: the collar off the dog, open, lying on the table beside it, both hands away.

The buckle works by contact only (mechkit.py, next to this file). The female half is the base of the collar's
articulation, whose 13 webbing links hang from it on hinges along x; the male half is a second small articulation,
glued to the last link at its webbing bar by an unbreakable mechkit bond (a FixedJoint; between two articulations
such a bond has been seen to pass no turning torque, so the bar may act as a pivot, as the webbing loop round a real
bar does; either way suits the task). The male's two prongs turn on vertical hinges with
springs (2.5 N at the tab when pressed flush); each prong's hook sits behind a rib inside the female, so the male
comes out only while both tabs are pressed in. No rule and no flag: the score reads poses only.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
object_jitter must be 0: the bond is authored at the declared spawn poses.
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
import offline_collar  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (rolled grips for the demo: wraps yam_arena.demo.grip_rotation)
from scoring import COLLAR, DOG, MALE  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # on top of the scorer's own 1 s window

INSTRUCTION = ("Take the collar off the dog manikin: unclip the buckle, draw the collar off its neck, lay it on the "
               "table beside the manikin and let go of it.")


# -- the parts --------------------------------------------------------------------------------------------------------
DOG_PART = mk.assembly(DOG, origin=L.DOG_ORIGIN, shapes=P.dog_shapes(), mass=L.DOG_MASS, color=L.DOG_TAN,
                       friction=L.DOG_FRICTION)
COLLAR_PART = mk.assembly(COLLAR, origin=L.B_ORIGIN, shapes=P.female_shapes(), mass=L.F_MASS, joints=P.link_joints(),
                          color=L.F_GREY, friction=L.COLLAR_FRICTION, tight_fit=True)
MALE_PART = mk.assembly(MALE, origin=L.B_ORIGIN, shapes=P.male_shapes(), mass=L.M_MASS,
                        joints=[P.prong(1.0), P.prong(-1.0)], color=L.BLACK, friction=L.COLLAR_FRICTION,
                        tight_fit=True)
# The webbing's end round the male's bar: glued, unbreakable; a pivot (see the docstring). The pair's turn is not a
# break, so the geometric slip check is set wide.
BAR_BOND = mk.bond("male_bar", MALE_PART, (COLLAR_PART, f"w{L.N_LINKS}"), anchor=(0.0, L.M_Y0, L.END_Z),
                   slip_mm=10.0, slip_deg=180.0)
OBJECTS = [DOG_PART, COLLAR_PART, MALE_PART]
SPECS = {COLLAR: COLLAR_PART, MALE: MALE_PART, DOG: DOG_PART}

_OFFLINE = offline_collar.CollarModel(COLLAR_PART, MALE_PART, DOG_PART)


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(COLLAR), "source", "sim") != "sim"


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
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Apart, off the dog, resting and hands off on every step of the last 1 s, no latch. This runs first every
    control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
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
        parts = _OFFLINE.states(S.PARTS)
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row["prongs"] = parts[MALE].joints
        row["female_origin"] = parts[COLLAR].origin
        row["male_origin"] = parts[MALE].origin
        row["dog_origin"] = parts[DOG].origin
        row["bond"] = BAR_BOND.status()
    except (AttributeError, KeyError, TypeError):
        pass
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(COLLAR), "source", None)
    if offline:
        col = parts[COLLAR]
        low = {b: round(float(np.min(col.link_to_world(b, S.points_of(col.spec, b))[:, 2]) - L.TOP) * 1000, 1)
               for b in col.spec.body_names}
        row["lowest_body"] = min(low, key=low.get)
        row["nodes_z_mm"] = [round((float(z) - L.TOP) * 1000, 1) for z in _OFFLINE.nodes[:, 2]]
        row["offline"] = {"mode": _OFFLINE.mode, "holds": {a: p for a, (p, _) in _OFFLINE.holds.items()},
                          "slips": _OFFLINE.slips}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the collar left the table or the manikin was dragged 100 mm, or the buckle is still clicked (male under
    3 mm out); 0.25 unclipped, not apart; 0.5 apart, still on or round the dog; 0.75 apart and off the dog, but not
    resting with both hands away for the whole last second, or the dog nudged 20 mm at some step; 1.0 all of it. The
    latest scored step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both grips close along
# world x (yaw -pi/2 left, +pi/2 right) and are rolled about x (tiltgrip.py) so the wrists lean apart: the left 0.35
# rad (pad tips toward -y), the right 0.85 rad (tips toward +y). Points below are the pad-tip points (table frame);
# _w turns them into TCP targets 13 mm back along the approach axis.
YAW_L, YAW_R = -math.pi / 2, math.pi / 2
TILT_L = tiltgrip.tilted(YAW_L, L.ROLL_L)
TILT_R = tiltgrip.tilted(YAW_R, L.ROLL_R)
A_L = tiltgrip.approach(YAW_L, L.ROLL_L)
A_R = tiltgrip.approach(YAW_R, L.ROLL_R)
PAD_TIP = 0.013


def fingers(gap: float) -> float:
    return -gap / 2


F_OPEN = fingers(L.OPEN_GAP_FEMALE)
F_PINCH = fingers(L.PINCH_GAP)
M_OPEN = fingers(L.OPEN_GAP_MALE)
M_GRIP = fingers(L.MALE_GAP)

# The left pad-tip point at the tabs' middle (y 10 mm in from the buckle frame's origin), 2 mm over the buckle's
# underside: its pads cover both tabs (3.5 to 7.5 mm up) and the housing; their lowest corner stays 2.6 mm over the
# neck.
PINCH = (L.BX, 0.010, L.BZ0 + 0.002)
# The right pad-tip point on the male's body (y -20 mm), 5 mm over the buckle's underside: the pads cover the body's
# side from y -26 to -6.5 mm (2.5 mm short of the female's mouth); lowest corner 3.5 mm over the neck.
GRIP = (L.BX, -0.020, L.BZ0 + 0.005)
PULL = 0.040  # the male out 40 mm: no part of it inside the female, its hook tips 13.7 mm clear of the mouth
DRAW_Z = 0.022  # dragging height of the right pad tips: the male's underside 17.5 mm over the table
END_Y = -0.445  # the male laid here: the webbing straight back along +y, the female's far end at y -96 mm
LAY_Z = 0.012
HOVER = 0.050  # over the buckle before the descent
PARK_L = (0.220, 0.200, 0.160)
PARK_R = (0.220, -0.250, 0.160)
# The right wrist never changes its yaw once it has the male: the pinch closes along world x and the webbing's 13
# hinges all turn about the female's x, so the male's yaw sets the open strap's plane. A yawed wrist swings that plane
# and steers the female, trailing 300 mm behind, sideways along the table (GPU run 20261119T034831: the wrist turned
# 0.35 rad as it went out past the neck, the female slid 66 mm toward +x past the neck's end at x 380 and jammed on the
# flank's corner, still on the far side of the neck). Only the roll about world x changes (it turns the male about its
# own pinch axis, leaving the strap's plane alone): from 0.85 rad, leaning away from the dog, it eases to 0.45 rad by y
# MID_Y and to 0 (straight down) by TURN_Y, which keeps the wrist joint inside its limit far out at y END_Y.
MID_Y = -0.205
TURN_Y = -0.280
ROLL_MID = 0.45
YAW_DRAG = YAW_R
TILT_MID = tiltgrip.tilted(YAW_DRAG, ROLL_MID)
A_MID = tiltgrip.approach(YAW_DRAG, ROLL_MID)
TILT_DRAG = tiltgrip.tilted(YAW_DRAG, 0.0)
A_DRAG = tiltgrip.approach(YAW_DRAG, 0.0)
DRAW_SPEED = 0.040  # m/s along -y


def _w(d, tip, finger, label, arm):
    tilt, a = {"left": (TILT_L, A_L), "right": (TILT_R, A_R), "mid": (TILT_MID, A_MID),
                "drag": (TILT_DRAG, A_DRAG)}[arm]
    t = np.array([tip[0], tip[1], L.TOP + tip[2]]) - PAD_TIP * a
    return Waypoint(d, (float(t[0]), float(t[1]), float(t[2])), finger, label, tilt)


def _park(d, xyz, finger, label, arm):
    tilt = {"left": TILT_L, "right": TILT_R, "drag": TILT_DRAG}[arm]
    return Waypoint(d, (xyz[0], xyz[1], L.TOP + xyz[2]), finger, label, tilt)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def _hold(plan, until, label):
    last = plan[-1]
    gap = until - _duration(plan)
    if gap > 1e-9:
        plan.append(Waypoint(gap, last.pos, last.finger, label, last.yaw))


def left_moves(release_at: float | None = None) -> list:
    """Over the female, down either side of it, pinch it across its width (the pads press both tabs flush and stop on
    the housing), hold it while the right hand pulls the male out, open, up, aside."""
    x, y, z = PINCH
    plan = [
        _w(2.5, (x, y, z + HOVER), F_OPEN, "over the female, fingers 44 mm apart", "left"),
        _w(1.5, (x, y, z), F_OPEN, "down either side of the female", "left"),
        _w(0.5, (x, y, z), F_OPEN, "settle", "left"),
        _w(1.5, (x, y, z), F_PINCH, "pinch across the female: both tabs pressed flush", "left"),
        _w(0.5, (x, y, z), F_PINCH, "settle the pinch", "left"),
    ]
    _hold(plan, release_at if release_at is not None else 13.0, "hold the tabs in while the male comes out")
    plan += [
        _w(1.5, (x, y, z), F_OPEN, "open", "left"),
        _w(1.0, (x, y, z + HOVER), F_OPEN, "straight up", "left"),
        _park(2.0, PARK_L, F_OPEN, "aside", "left"),
    ]
    return plan


def right_moves(end_y: float = END_Y, start_at: float = 6.5, draw_at: float = 15.5) -> list:
    """Over the male, down either side of it once the left hand has the tabs in, grip it, pull it 40 mm straight
    out; when the left hand is clear, out past the neck's side, down to the table, drag the open collar round and
    out from under the neck along the table, lay the male down, open, up, aside."""
    x, y, z = GRIP
    plan = [_w(3.0, (x, y, z + HOVER), M_OPEN, "over the male, fingers 36 mm apart", "right")]
    _hold(plan, start_at, "wait for the left pinch")
    plan += [
        _w(1.5, (x, y, z), M_OPEN, "down either side of the male", "right"),
        _w(0.5, (x, y, z), M_OPEN, "settle", "right"),
        _w(1.5, (x, y, z), M_GRIP, "grip the male's body", "right"),
        _w(0.5, (x, y, z), M_GRIP, "settle the grip", "right"),
        _w(2.0, (x, y - PULL, z), M_GRIP, "pull the male 40 mm straight out of the female", "right"),
        _w(0.5, (x, y - PULL, z), M_GRIP, "settle", "right"),
    ]
    _hold(plan, draw_at, "wait for the left hand to clear")
    plan += [
        _w(1.5, (x, -0.110, z), M_GRIP, "out past the neck's side, the wrist's yaw unchanged", "right"),
        _w(2.0, (x, -0.120, 0.060), M_GRIP, "down beside the neck", "right"),
        _w(1.5, (x, -0.130, DRAW_Z), M_GRIP, "down to just over the table", "right"),
        _w((MID_Y - (-0.130)) / -DRAW_SPEED, (x, MID_Y, DRAW_Z), M_GRIP, "draw the collar round and out from under "
           "the neck, about 40 mm/s, the wrist rolling upright", "mid"),
        _w((TURN_Y - MID_Y) / -DRAW_SPEED, (x, TURN_Y, DRAW_Z), M_GRIP, "on under the neck, the wrist rolling "
           "straight down", "drag"),
        _w((end_y - TURN_Y) / -DRAW_SPEED, (x, end_y, DRAW_Z), M_GRIP, "on along the table, the female out from "
           "under the neck", "drag"),
        _w(1.0, (x, end_y, DRAW_Z), M_GRIP, "settle", "drag"),
        _w(1.0, (x, end_y, LAY_Z), M_GRIP, "lower the male onto the table", "drag"),
        _w(0.5, (x, end_y, LAY_Z), M_GRIP, "settle", "drag"),
        _w(1.5, (x, end_y, LAY_Z), M_OPEN, "let go", "drag"),
        _w(0.5, (x, end_y, LAY_Z), M_OPEN, "settle", "drag"),
        _w(1.0, (x, end_y, LAY_Z + 0.060), M_OPEN, "straight up", "drag"),
        _park(2.0, PARK_R, M_OPEN, "aside", "drag"),
    ]
    return plan


# Wrong ways (counterfactuals; the offline model predicts them):
#   early   the left lets go of the tabs before the right pulls: the hooks drop back behind their ribs, the pull
#           drags the whole collar and the hand slips off: 0
#   short   the collar drawn only to y -250 mm: the female still under or beside the neck: 0.5
VARIANTS = ("early", "short")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "early":
        return {"left": left_moves(release_at=9.0), "right": right_moves()}
    if variant == "short":
        return {"left": left_moves(), "right": right_moves(end_y=-0.250)}
    return {"left": left_moves(), "right": right_moves()}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check). DEMO_VARIANT: see VARIANTS."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
