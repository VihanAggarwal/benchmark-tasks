"""On the station table: a pale blue square glass (74 x 74 mm, 80 mm tall, open at the top, 62 x 62 mm inside) that
is evidence with fingerprints on its outside, standing on a yellow start mark 38 cm left of the midline, in front of
the left arm; a white exam sheet (140 x 140 mm) on the midline; and a brown evidence box (four fixed walls 55 mm
tall round a 105 x 105 mm floor) 38 cm right of the midline, in front of the right arm. The glass must go to the
exam sheet with the left gripper, then into the box with the right gripper, gripped only from the inside: the
fingers go in nearly closed and open until they press its inner walls. No finger may be on its outside, its
outside may touch nothing (not the box's walls), and it may be set down only on its mark, the sheet or the box
floor, never tipped or dropped.

The glass is a mechkit assembly (mechkit.py, next to this file: one loose rigid body made of a floor and four
walls); the scorer reads the arms (tool point, finger tips, opening) through rigstate.py, which wraps the
embodiment's step and reset from this file's import. Neither changes what the embodiment does. offline_model.py is
used only by our offline checker (no physics there): the simulator never calls it.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import numpy as np  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the glass is built as one compound body)
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

_RIG_HOOKED = rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it)
if not _RIG_HOOKED and not getattr(sys.modules.get("yam_arena"), "OFFLINE_STUB", False):
    print("[latent-print-glass-inside] WARNING: rigstate could not hook yam_arena's embodiment, so the arms are "
          "read only after each step through mechkit's env; if they cannot be read on more than 1% of the steps "
          "the trial scores 0 (the finger rule cannot be judged)", file=sys.stderr)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s in the box with both grippers let go (tips 20 mm above the rim or 60 mm from the glass's axis): the
# trial cannot end while a gripper is still inside the glass.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "The glass is evidence with fingerprints on its outside, so touch it only on the inside: put a gripper's fingers "
    "into it nearly closed, then open them against its inner walls. With the left gripper move it from its yellow "
    "mark onto the white exam sheet, then with the right gripper into the brown evidence box, standing on the box "
    "floor, and let go. Never let its outside touch anything, and set it down only on its mark, the sheet or the "
    "box floor."
)

# -- the objects --------------------------------------------------------------------------------------------------
GLASS = mk.assembly(
    "glass", origin=(L.START_X, L.START_Y, L.MARK_TOP), mass=L.GLASS_MASS, color=L.GLASS_BLUE,
    shapes=mk.open_box((0.0, 0.0, L.GLASS_HALF_H), (L.GLASS_OUTER, L.GLASS_OUTER, L.GLASS_H), wall=L.GLASS_WALL,
                       floor=L.GLASS_FLOOR),
)
START_MARK = Box("start_mark", pos=(L.START_X, L.START_Y, L.TOP + L.MARK_SIZE[2] / 2), size=L.MARK_SIZE,
                 color=L.MARK_YELLOW)
EXAM_SHEET = Box("exam_sheet", pos=(L.SHEET_X, L.SHEET_Y, L.TOP + L.SHEET_SIZE[2] / 2), size=L.SHEET_SIZE,
                 color=L.SHEET_WHITE)


def _wall(name: str, r) -> Box:
    x0, x1, y0, y1 = r
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.BOX_HEIGHT / 2),
               size=(x1 - x0, y1 - y0, L.BOX_HEIGHT), color=L.BOX_BROWN)


BOX_WALLS = [_wall(f"box_{side}", r) for side, r in S.BOX_WALLS.items()]  # the table top is the box's floor
OBJECTS = [START_MARK, EXAM_SHEET, *BOX_WALLS, GLASS]

_TRACKER = S.Tracker()
_MODEL = offline_model.GlassModel()
_EPISODE = {"mechkit": None}
_ARMS = {"step": None, "arms": None, "how": None}
_WARNED: set = set()


def _warn_once(key: str, text: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[latent-print-glass-inside] {text}", file=sys.stderr)


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    _MODEL.reset()
    _ARMS.update(step=None, arms=None, how=None)


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _carried_along(arms, glass):
    """A reading taken before the physics step, moved one step on for each gripper that is inside the glass (it moves
    with it): its tool point and tips shifted by the glass's velocity x one control step."""
    if not S.arms_valid(arms) or glass is None or not S.valid(glass):
        return arms
    shift = np.asarray(glass.lin_vel, dtype=float) / L.CONTROL_HZ
    out = dict(arms)
    for side in S.ARMS:
        a = arms[side]
        moved = rigstate.Arm(tcp=np.asarray(a.tcp, dtype=float) + shift, tips=np.asarray(a.tips, dtype=float) + shift,
                             opening=a.opening)
        if S.tips_inside(moved, glass):
            out[side] = moved
    return out


def _sim_arms(glass):
    """Both arms at the same moment as the objects. rigstate takes its snapshot BEFORE the embodiment steps the
    physics (one control step older than the object states the hooks get), so read them again now, with rigstate's
    own reader on the env that mechkit keeps (both files unchanged). If that fails, fall back to rigstate's snapshot
    carried one step on with the glass for a gripper inside it. Read once per control step (mechkit's counter)."""
    step = mk.step()
    if step and _ARMS["step"] == step:
        return _ARMS["arms"], _ARMS["how"]
    arms, how = None, None
    env = mk.env()
    if env is not None:
        try:
            arms, how = rigstate._snapshot(env), "after_step"
        except Exception as exc:  # noqa: BLE001 - reported, then the older snapshot is used
            _warn_once("fresh", f"could not read the arms after the step ({exc!r}); using rigstate's snapshot from "
                                f"before it, moved on with the glass for a gripper inside it")
    if arms is None:
        arms = rigstate.arms()
        how = "before_step" if arms is not None else None
        if arms is None:
            _warn_once("none", f"the arms cannot be read ({rigstate.error()!r}): the finger rule is not applied on "
                               f"such steps, and if they are more than 1% of the steps the trial scores 0")
        else:
            arms = _carried_along(arms, glass)
    _ARMS.update(step=step, arms=arms, how=how)
    return arms, how


def _observe(objects):
    """The glass and the arms the scorer judges this step. In the simulator: the glass's state and the arms read at
    the same moment (_sim_arms). Offline (tools/offline/check_task.py: no physics, the glass never moves by itself
    there, and its state says source "offline") the checker feeds rigstate the arms for this very step, and the glass
    and the arms come from offline_model: where an internal grip would carry it. The simulator never takes that
    branch (its states say source "sim")."""
    g = objects.get("glass") if hasattr(objects, "get") else None
    if getattr(g, "source", "sim") == "offline":
        glass, seen = _MODEL.step(g, rigstate.arms())
        return glass, seen, "offline"
    arms, how = _sim_arms(g)
    return g, arms, how


def success(objects) -> bool:
    """In the box (wholly on its floor, upright, still), both grippers let go, stood on the exam sheet earlier,
    no rule ever broken; held 1 s. This runs first every control step, so it also feeds the tracker (with both arms
    read at the same moment as the glass)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _MODEL.reset()
            _ARMS.update(step=None, arms=None, how=None)
        _EPISODE["mechkit"] = mk.episode()
    glass, arms, how = _observe(objects)
    offline = how == "offline"
    result = _TRACKER.update(glass, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "source": getattr(glass, "source", None), "arms_read": how}
    try:
        row["glass_pos"] = glass.pos
        row["glass_vel"] = glass.lin_vel
    except AttributeError:
        row["state_error"] = True
    if S.arms_valid(arms):
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    if offline:
        row["model_holder"] = _MODEL.holder
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if at any moment a finger was on the glass's outside (or within 1 cm of it below the rim), the glass
    touched a box wall, tilted more than 10 deg, was set down off its mark, the sheet and the box floor, was dropped,
    or left the table, or if the arms could not be read on more than 1% of the steps. Else 1.0 standing in the box with both grippers let go, having stood on the exam sheet;
    0.75 standing in the box otherwise; 0.5 standing on the exam sheet; 0.25 lifted 20 mm at some moment; 0."""
    glass, arms, _ = _observe(objects)
    return _TRACKER.result(glass, arms).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Yaw 0 throughout (fingers closing along y; the glass is square). Each grip: over the glass with the fingers
# nearly closed (10 mm apart), down until the tips are 10 mm above the rim, then 52 mm more over 2 s (tips 42 mm
# below the rim), settle, open to 50 mm over 1.5 s (the outer faces meet the walls at 42 mm: 4 mm more per finger,
# about 20 N each), settle, lift. Each set-down: over the target, down until the glass's bottom is 15 mm above it,
# settle, the last part over 1.5 s, settle, close to 10 mm over 1.5 s (off the walls), settle, straight up.
# Wrist sag under the glass's 1.8 N (yamkin, joints 4-6 at 10 N m/rad, the payload not compensated): about 4.7 mm
# down at the exam sheet, 2.6 mm at the box, 1.5-2 mm sideways, under 1 deg of tilt; so the glass is lowered to
# 3 mm above the sheet and 1.5 mm above the box floor as commanded and comes to rest on them before the fingers
# close. On the GPU, read low_mm and max_fall_mps in TASK_TRACE at each release and tune only SET_GAP_*.
OPEN = -0.04695
NEAR_CLOSED = -L.NEAR_CLOSED_GAP / 2  # -0.005: fingers 10 mm apart, 16 mm clear of each inner wall
GRIP = -L.GRIP_GAP / 2  # -0.025: fingers commanded 50 mm apart; the inner walls stop them at 42 mm
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
TIP_DEPTH = 0.042  # tips this far below the rim while gripping: the measured lowest 38 mm of the fingers inside
HOLD = L.GLASS_H - TIP_DEPTH + PAD_TIPS  # 0.051: the TCP is this far above the glass's bottom face while gripping
ENTRY = L.GLASS_H + 0.010 + PAD_TIPS  # 0.103: TCP above the glass's bottom with the tips 10 mm above the rim
YAW = 0.0
HOVER_L = L.TOP + 0.140
HOVER_R = L.TOP + 0.140
CARRY_L = L.TOP + 0.126  # the glass's bottom 75 mm up on the way to the sheet
CARRY_R = L.TOP + 0.136  # the glass's bottom 85 mm up: 30 mm over the box's walls
SET_LOW = 0.015  # the slow last part of each set-down starts with the glass's bottom this far above its surface
SET_GAP_SHEET = 0.003  # commanded height of the glass's bottom over the sheet at release (sag takes it down)
SET_GAP_BOX = 0.0015  # ... over the box floor
WAIT_R = (L.SHEET_X, -0.24)  # where the right arm waits: 24 cm right of the left arm working on the sheet
RETREAT_L = (0.22, 0.25, HOVER_L)  # where the left arm goes once it has let go
STAGGER = 0.5  # the right arm sets off for the sheet this long after the left arm starts back


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _grip(x: float, y: float, bottom: float, what: str) -> list[Waypoint]:
    """Go in through the mouth nearly closed and open against the inner walls; ``bottom`` is the glass's bottom."""
    return [
        Waypoint(1.5, (x, y, bottom + ENTRY), NEAR_CLOSED, f"down over {what}: tips 10 mm above the rim", YAW),
        Waypoint(2.0, (x, y, bottom + HOLD), NEAR_CLOSED, "into the glass, slowly: tips 42 mm below the rim", YAW),
        Waypoint(0.6, (x, y, bottom + HOLD), NEAR_CLOSED, "settle", YAW),
        Waypoint(1.5, (x, y, bottom + HOLD), GRIP, "open against the inner walls", YAW),
        Waypoint(0.6, (x, y, bottom + HOLD), GRIP, "settle", YAW),
    ]


def _set_down(x: float, y: float, surface: float, gap: float, what: str) -> list[Waypoint]:
    """Lower the glass onto ``surface`` (its top's height), let go from inside and come straight up."""
    return [
        Waypoint(1.2, (x, y, surface + SET_LOW + HOLD), GRIP, f"down: the glass 15 mm above {what}", YAW),
        Waypoint(0.4, (x, y, surface + SET_LOW + HOLD), GRIP, "settle", YAW),
        Waypoint(1.5, (x, y, surface + gap + HOLD), GRIP, f"lower it onto {what}, slowly", YAW),
        Waypoint(0.6, (x, y, surface + gap + HOLD), GRIP, "settle", YAW),
        Waypoint(1.5, (x, y, surface + gap + HOLD), NEAR_CLOSED, "let go: fingers off the walls, 10 mm apart", YAW),
        Waypoint(0.3, (x, y, surface + gap + HOLD), NEAR_CLOSED, "settle", YAW),
    ]


def _left() -> list[Waypoint]:
    """Left arm: grip the glass from inside on its mark, carry it to the exam sheet, set it down, let go, come up
    and move back toward its own side."""
    sx, sy = L.START_X, L.START_Y
    ex, ey = L.SHEET_X, L.SHEET_Y
    return [
        Waypoint(2.0, (sx, sy, HOVER_L), NEAR_CLOSED, "over the glass, closing the fingers to 10 mm", YAW),
        *_grip(sx, sy, L.MARK_TOP, "the glass on its mark"),
        Waypoint(1.5, (sx, sy, CARRY_L), GRIP, "lift it straight up", YAW),
        Waypoint(2.5, (ex, ey, CARRY_L), GRIP, "carry it over the exam sheet", YAW),
        *_set_down(ex, ey, L.SHEET_TOP, SET_GAP_SHEET, "the exam sheet"),
        Waypoint(1.5, (ex, ey, HOVER_L), NEAR_CLOSED, "straight up, out of the glass", YAW),
        Waypoint(1.5, RETREAT_L, NEAR_CLOSED, "back toward its own side", YAW),
    ]


def _right() -> list[Waypoint]:
    """Right arm: wait 24 cm to the right of the sheet while the left arm works; then grip the glass from inside on
    the sheet, carry it over the evidence box, lower it onto the box floor, let go and come up."""
    left = _left()
    t_go = _span(left[:-1]) + STAGGER  # the left arm is out of the glass, 14 cm up, and on its way back
    bx, by = L.BOX_X, L.BOX_Y
    ex, ey = L.SHEET_X, L.SHEET_Y
    plan = [Waypoint(3.0, (*WAIT_R, HOVER_R), NEAR_CLOSED, "wait to the right of the sheet, fingers to 10 mm", YAW)]
    plan.append(Waypoint(t_go - _span(plan), (*WAIT_R, HOVER_R), NEAR_CLOSED,
                         "wait while the left arm works on the sheet", YAW))
    plan += [
        Waypoint(1.5, (ex, ey, HOVER_R), NEAR_CLOSED, "over the glass as the left arm moves away", YAW),
        *_grip(ex, ey, L.SHEET_TOP, "the glass on the sheet"),
        Waypoint(1.5, (ex, ey, CARRY_R), GRIP, "lift it straight up, its bottom 30 mm over the box's walls", YAW),
        Waypoint(2.5, (bx, by, CARRY_R), GRIP, "carry it over the evidence box", YAW),
        *_set_down(bx, by, L.TOP, SET_GAP_BOX, "the box floor"),
        Waypoint(1.5, (bx, by, HOVER_R), NEAR_CLOSED, "straight up, out of the glass and the box", YAW),
    ]
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
