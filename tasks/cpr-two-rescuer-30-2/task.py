"""On the station table: a training manikin, fixed in place, lying along y with its head toward the robot's left: a
skin-coloured torso shell (170 x 300 x 80 mm) 21.5 to 38.5 cm in front of the arm bases, just right of the midline,
whose chest is a plate (115 x 115 mm) on a vertical spring (75 N/m, 65 mm of travel) with a dark mark at its
centre, level with the torso's top; a neck and a head (100 x 90 x 70 mm). Beyond the head, 22 cm left of the
midline, stands the 'bag': a blue cap (60 x 60 mm, 60 mm tall, top 11 cm above the table) on a vertical spring
(40 N/m) over a dark post, standing in for a bag-valve-mask's bag. Two-rescuer CPR: two cycles of 30 compressions
(5-6 cm deep, 100-120 a minute, full recoil) and 2 breaths (about 1 s each, given by one gripper pressing the cap,
the chest not pressed meanwhile and the other gripper waiting over the mark), pauses under 10 s. One arm cannot both
wait over the mark and press the bag; with a straight-down gripper the bag is out of the right arm's reach and the
left arm's tool point stays 5 cm or more from the mark.

The manikin is a mechkit assembly (mechkit.py, next to this file: a fixed-base articulation with two prismatic
joints); the scorer reads the joints from it, and the arms (tool point, finger tips, opening) through rigstate.py,
which wraps the embodiment's step and reset from this file's import. Neither changes what the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the manikin is built as an articulation)
import numpy as np  # noqa: E402
import offline_press  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s once both grippers are 30 mm or more from the chest plate and the cap: the trial cannot end while
# a gripper still rests on either.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Give two cycles of two-rescuer CPR on the manikin. In each cycle press the dark mark on its chest 30 times, 5 to "
    "6 cm deep, 100 to 120 times a minute, letting the chest come all the way back up each time; then give 2 breaths, "
    "each by pushing the blue bag by its head about 3 cm down for about 1 second, while the other gripper waits over "
    "the mark without pressing. Keep each pause in compressions under 10 seconds. Then lift both grippers 3 cm clear "
    "of the chest and the bag."
)

# -- the manikin: one fixed assembly; shapes in its base frame (origin L.ORIGIN, on the table top) ---------------
OX, OY = L.ORIGIN[0], L.ORIGIN[1]


def _local(x: float, y: float, z: float) -> tuple[float, float, float]:
    """A world (x, y) and a height above the table, in the manikin's base frame."""
    return (x - OX, y - OY, z)


_TX, _TY, _TZ = L.TORSO_SIZE
_WALL_H = _TZ - L.DECK
TORSO = [
    mk.box(_local(OX + _TX / 2 - L.WALL / 2, L.TORSO_Y, _WALL_H / 2), (L.WALL, _TY, _WALL_H)),  # far side
    mk.box(_local(OX - _TX / 2 + L.WALL / 2, L.TORSO_Y, _WALL_H / 2), (L.WALL, _TY, _WALL_H)),  # near side
    mk.box(_local(OX, L.TORSO_Y + _TY / 2 - L.WALL / 2, _WALL_H / 2), (_TX - 2 * L.WALL, L.WALL, _WALL_H)),  # shoulders
    mk.box(_local(OX, L.TORSO_Y - _TY / 2 + L.WALL / 2, _WALL_H / 2), (_TX - 2 * L.WALL, L.WALL, _WALL_H)),  # hips
    *mk.plate_with_hole(_local(OX, L.TORSO_Y, _TZ - L.DECK / 2), (_TX, _TY), (L.OPENING, L.OPENING), L.DECK,
                        hole_offset=(L.MARK_X - OX, L.MARK_Y - L.TORSO_Y)),  # the top deck with the chest opening
]
NECK_AND_HEAD = [
    mk.box(_local(OX, L.NECK_Y, L.NECK_SIZE[2] / 2), L.NECK_SIZE),
    mk.box(_local(OX, L.HEAD_Y, L.HEAD_SIZE[2] / 2), L.HEAD_SIZE),
]
POST = mk.box(_local(L.BAG_X, L.BAG_Y, L.POST_SIZE[2] / 2), L.POST_SIZE, color=L.POST_GREY)

# The chest plate, drawn with its top level with the deck; positive travel pushes it down (a compression).
CHEST = mk.slider(
    "chest", "-Z",
    [mk.box(_local(L.MARK_X, L.MARK_Y, L.PLATE_TOP - L.PLATE_SIZE[2] / 2), L.PLATE_SIZE),
     mk.cylinder(_local(L.MARK_X, L.MARK_Y, L.PLATE_TOP + L.MARK_THICK / 2), L.MARK_RADIUS, L.MARK_THICK,
                 color=L.MARK_COLOR)],
    upper=L.CHEST_TRAVEL, spring=L.CHEST_SPRING, damping=L.CHEST_DAMPING, mass=L.PLATE_MASS, color=L.SKIN,
)
# The bag's cap: a sleeve (top plate and four side walls) that slides down over the post when its top is pressed.
_CW, _CH, _CP, _CT = L.CAP_SIZE[0], L.CAP_SIZE[2], L.CAP_PLATE, L.CAP_WALL
_SKIRT_H = _CH - _CP
_SKIRT_Z = L.CAP_TOP - _CP - _SKIRT_H / 2
BAG = mk.slider(
    "bag", "-Z",
    [mk.box(_local(L.BAG_X, L.BAG_Y, L.CAP_TOP - _CP / 2), (_CW, _CW, _CP)),
     mk.box(_local(L.BAG_X + (_CW - _CT) / 2, L.BAG_Y, _SKIRT_Z), (_CT, _CW, _SKIRT_H)),
     mk.box(_local(L.BAG_X - (_CW - _CT) / 2, L.BAG_Y, _SKIRT_Z), (_CT, _CW, _SKIRT_H)),
     mk.box(_local(L.BAG_X, L.BAG_Y + (_CW - _CT) / 2, _SKIRT_Z), (_CW - 2 * _CT, _CT, _SKIRT_H)),
     mk.box(_local(L.BAG_X, L.BAG_Y - (_CW - _CT) / 2, _SKIRT_Z), (_CW - 2 * _CT, _CT, _SKIRT_H))],
    upper=L.BAG_TRAVEL, spring=L.BAG_SPRING, damping=L.BAG_DAMPING, mass=L.CAP_MASS, color=L.CAP_BLUE,
)
MANIKIN = mk.assembly("manikin", origin=L.ORIGIN, shapes=[*TORSO, *NECK_AND_HEAD, POST], joints=[CHEST, BAG],
                      color=L.SKIN)
OBJECTS = [MANIKIN]

_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the arms jumping back to ready)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _joints(m, arms) -> dict:
    """The chest's and the cap's travel (m, positive pressed down). In the simulator: mechkit's joint readings
    (NaN if the manikin's state is broken). Offline (tools/offline/check_task.py: no physics, the joints stay at 0)
    the scorer is fed offline_press.kinematic_joints instead: where the finger tips would hold each slider. The
    simulator never takes that branch (its states say source "sim")."""
    if getattr(m, "source", "sim") == "offline":
        return offline_press.kinematic_joints(arms)
    try:
        if not np.all(np.isfinite(np.asarray(m.origin, dtype=float))):
            raise ValueError("origin")
        return {"chest": float(m.joints["chest"]), "bag": float(m.joints["bag"])}
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"chest": math.nan, "bag": math.nan}


def success(objects) -> bool:
    """Two cycles of 28-32 compressions at 100-120 a minute, at least 90 % of them good, exactly 2 counted breaths
    after each cycle and none before or during one, both pauses within 10 s; both grippers clear of the plate and
    the cap and the manikin still (held 1 s). This runs first every control step, so it also feeds the tracker
    (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    m = objects["manikin"]
    arms = rigstate.arms()
    joints = _joints(m, arms)
    result = _TRACKER.update(joints, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "source": getattr(m, "source", None)}
    try:
        row.update(joint_vel={k: float(v) for k, v in m.joint_vel.items()}, origin=m.origin)
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the chest was ever pressed past 20 mm while the cap was past 15 mm, or with fewer than 20 good
    compressions. Else 1.0 for exactly 2 cycles of 28-32 at 100-120 a minute, >= 90 % good, exactly 2 counted
    breaths after each cycle and none before or during one, pauses <= 10 s, and both grippers 30 mm clear; 0.75 for
    2 or more cycles, the first two of 25-35, >= 80 % good, >= 3 counted breaths in their pauses, those pauses
    <= 10 s; 0.5 for a cycle of 25-35 followed by a counted breath, >= 60 % good, every pause <= 15 s; else 0.25."""
    arms = rigstate.arms()
    return _TRACKER.result(_joints(objects["manikin"], arms), arms).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Both arms close their fingers on the way to their stations and push with the closed finger tips (13 mm below the
# tool point). The right arm's fingers close along the diagonal (yaw 45 deg): at the bottom of a stroke the closed
# finger carriages (48 mm either side of the gripper's axis, 57-81 mm above the tool point) then stand over the
# 120 x 120 mm chest opening's diagonal, 34 mm from its centre each way; the gripper's and fingers' collision capsules
# stay at least 8 mm clear of the deck (offline, without the arm's give). The left arm keeps yaw 0 (fingers closing
# along y). Both keep every wrist joint more than 10 deg from its limits.
OPEN, CLOSED = -0.04695, 0.0
YAW_R = math.pi / 4
YAW_L = 0.0
PAD_TIPS = 0.013
# Right arm (compressions). The tips meet the mark at tool point z 0.094 as drawn. The top of each stroke leaves them
# 6 mm above it (full recoil: the plate rests 2 mm down under its own weight), the tool point 19 mm above the mark:
# there it waits during the breaths, in place over the mark as the breath rule asks. At the bottom the arm gives about
# 1.9 mm per newton against the 4 N spring, so a stroke to z 0.031 holds the chest about 55 mm down (offline,
# static). On the GPU read depth_mm / recoil_mm in TASK_TRACE (last_compression) and tune only STROKE_BOTTOM (and the
# 0.25 / 0.30 s split) until the depths land at 55 +- 3 mm: arm lag at 1.8 strokes a second makes them shallower. If
# recoil_mm reaches 4, raise STROKE_TOP (at most TOP + 0.125: it is also where the arm waits over the mark).
CHEST_HOVER = L.TOP + 0.130
STROKE_TOP = L.TOP + 0.100
STROKE_BOTTOM = L.TOP + 0.031
DOWN_S, UP_S = 0.25, 0.30  # 0.55 s a compression: 109 a minute
COMPRESSIONS = 30
CYCLE_STARTS = (4.0, 24.4)
CHEST_CLEAR = L.TOP + 0.150
# Left arm (breaths). The tips meet the cap's top at tool point z 0.123; pressing to 0.085 holds it about 36 mm down
# (the breath rule wants a tool point within 40 mm above the cap's top at its deepest: here 13 mm).
BAG_HOVER = L.TOP + 0.140
BAG_PRESS = L.TOP + 0.085
BAG_CLEAR = L.TOP + 0.170
BREATH_DOWN_S, BREATH_HOLD_S, BREATH_UP_S, BREATH_GAP_S = 0.3, 0.7, 0.4, 0.3
# The breaths start 0.4 s after the right arm's last up-stroke ends; cycle 2 starts 0.4 s after the last breath's
# up-move ends. Offline, without lag, 0.65 s pass between the chest back under 10 mm and the cap past 15 mm, and 0.75 s
# between the cap back under 15 mm and the chest past 20 mm (the 0 rule needs both at once); pauses 4.45 and 3.6 s.
# If last_breath.chest_max_mm reads over 3 on the GPU, raise BREATH_DELAY_S.
BREATH_DELAY_S = 0.4


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _cycle(n: int = COMPRESSIONS) -> list[Waypoint]:
    p = (L.MARK_X, L.MARK_Y)
    out = []
    for i in range(n):
        out += [Waypoint(DOWN_S, (*p, STROKE_BOTTOM), CLOSED, f"compression {i + 1}: down", YAW_R),
                Waypoint(UP_S, (*p, STROKE_TOP), CLOSED, f"compression {i + 1}: up, full recoil", YAW_R)]
    return out


def _breaths() -> list[Waypoint]:
    p = (L.BAG_X, L.BAG_Y)
    out = []
    for i in (1, 2):
        out += [Waypoint(BREATH_DOWN_S, (*p, BAG_PRESS), CLOSED, f"breath {i}: press the bag", YAW_L),
                Waypoint(BREATH_HOLD_S, (*p, BAG_PRESS), CLOSED, f"breath {i}: hold it about a second", YAW_L),
                Waypoint(BREATH_UP_S, (*p, BAG_HOVER), CLOSED, f"breath {i}: let it come back up", YAW_L)]
        if i == 1:
            out.append(Waypoint(BREATH_GAP_S, (*p, BAG_HOVER), CLOSED, "between the breaths", YAW_L))
    return out


CYCLE_ENDS = tuple(t0 + COMPRESSIONS * (DOWN_S + UP_S) for t0 in CYCLE_STARTS)  # (20.5, 40.9): last up-stroke done
BREATHS_END = CYCLE_ENDS[-1] + BREATH_DELAY_S + _span(_breaths())  # 44.4: the second pair of breaths is done


def _right() -> list[Waypoint]:
    """Right arm: to the mark, closing the fingers; two cycles of 30 compressions, waiting over the mark (not
    pressing) while the other arm gives the breaths after each; then up, clear of the chest."""
    p = (L.MARK_X, L.MARK_Y)
    plan = [
        Waypoint(3.0, (*p, CHEST_HOVER), CLOSED, "over the mark, closing the fingers", YAW_R),
        Waypoint(CYCLE_STARTS[0] - 3.0, (*p, STROKE_TOP), CLOSED, "down to just above the mark", YAW_R),
    ]
    plan += _cycle()
    plan.append(Waypoint(CYCLE_STARTS[1] - _span(plan), (*p, STROKE_TOP), CLOSED,
                         "wait over the mark: the other arm gives 2 breaths", YAW_R))
    plan += _cycle()
    plan.append(Waypoint(BREATHS_END - _span(plan), (*p, STROKE_TOP), CLOSED,
                         "wait over the mark: the other arm gives 2 breaths", YAW_R))
    return plan + [Waypoint(1.5, (*p, CHEST_CLEAR), CLOSED, "up, off the chest", YAW_R)]


def _left() -> list[Waypoint]:
    """Left arm: over the bag, closing the fingers; after each cycle of compressions, 2 breaths of about a second;
    then up, clear of the bag."""
    p = (L.BAG_X, L.BAG_Y)
    plan = [Waypoint(3.0, (*p, BAG_HOVER), CLOSED, "over the bag, closing the fingers", YAW_L)]
    for end in CYCLE_ENDS:
        plan.append(Waypoint(end + BREATH_DELAY_S - _span(plan), (*p, BAG_HOVER), CLOSED,
                             "wait for the 30 compressions", YAW_L))
        plan += _breaths()
    return plan + [Waypoint(0.3, (*p, BAG_HOVER), CLOSED, "settle", YAW_L),
                   Waypoint(1.0, (*p, BAG_CLEAR), CLOSED, "up, off the bag", YAW_L)]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
