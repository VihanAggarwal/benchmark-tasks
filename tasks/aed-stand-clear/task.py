"""On the station table: an adult CPR manikin stand-in lying face up across the table, head toward the robot's left
(a fixed skin-coloured torso block, 240 mm across the chest, 300 mm long, 50 mm thick, and a head block), with a red
compression mark at the centre of the chest and two blue 40 mm pad squares (below the patient's right collarbone,
and on the patient's left side); a yellow-green AED (a fixed box) beside the head with an orange shock button on
top; and two loose white electrode pads (40 x 40 x 24 mm blocks) on the table beside the head. Two-rescuer rule:
the right gripper keeps chest compressions going while the left puts the pads on; both stand clear of the manikin
for the AED's 3 s analysis and while the shock button is pressed; compressions restart at once after the shock.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

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
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long with both pads still. Longer than a release and retreat (1.5 s + 1.0 s), so the trial
# cannot end with a gripper still on a pad; the demo's last release is long before success is possible anyway.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Do two-rescuer CPR with the AED on the manikin. With the right gripper closed, start chest compressions on the "
    "red mark within 10 seconds: repeated strokes from at least 3 cm above the mark down to within 1.5 cm of it, "
    "without pressing on the chest. Keep compressing while the left gripper puts one white pad flat on each blue "
    "square. Then both grippers stand clear, more than 5 cm from the manikin, while the AED analyses for 3 seconds; "
    "then press the orange shock button with the closed left gripper. Never touch the button while a gripper is "
    "near the manikin. Never pause compressions for more than 10 seconds before the shock; restart them within 3 "
    "seconds after it and keep them going."
)

TORSO = Box("manikin_torso", pos=(L.TORSO_X, L.TORSO_Y, L.TOP + L.TORSO_SIZE[2] / 2), size=L.TORSO_SIZE,
            color=L.SKIN)
HEAD = Box("manikin_head", pos=(L.HEAD_X, L.HEAD_Y, L.TOP + L.HEAD_SIZE[2] / 2), size=L.HEAD_SIZE, color=L.SKIN)
_mz = L.TORSO_TOP + L.MARK_THICK / 2
CPR_MARK = Box("compression_mark", pos=(L.CPR_X, L.CPR_Y, _mz), size=(L.CPR_MARK, L.CPR_MARK, L.MARK_THICK),
               color=L.RED)
STERNAL_MARK = Box("sternal_site", pos=(L.STERNAL_X, L.STERNAL_Y, _mz), size=(L.SITE_MARK, L.SITE_MARK, L.MARK_THICK),
                   color=L.BLUE)
APEX_MARK = Box("apex_site", pos=(L.APEX_X, L.APEX_Y, _mz), size=(L.SITE_MARK, L.SITE_MARK, L.MARK_THICK),
                color=L.BLUE)
AED = Box("aed", pos=(L.AED_X, L.AED_Y, L.TOP + L.AED_SIZE[2] / 2), size=L.AED_SIZE, color=L.AED_GREEN)
BUTTON = Cylinder("shock_button", pos=(L.AED_X, L.AED_Y, L.AED_TOP + L.BUTTON_HEIGHT / 2), radius=L.BUTTON_RADIUS,
                  height=L.BUTTON_HEIGHT, color=L.ORANGE)
PAD_A = Box("pad_a", pos=(L.PAD_A_X, L.PAD_A_Y, L.TOP + L.PAD_SIZE[2] / 2), size=L.PAD_SIZE, mass=L.PAD_MASS,
            color=L.WHITE)
PAD_B = Box("pad_b", pos=(L.PAD_B_X, L.PAD_B_Y, L.TOP + L.PAD_SIZE[2] / 2), size=L.PAD_SIZE, mass=L.PAD_MASS,
            color=L.WHITE)
OBJECTS = [TORSO, HEAD, CPR_MARK, STERNAL_MARK, APEX_MARK, AED, BUTTON, PAD_A, PAD_B]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
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


def success(objects) -> bool:
    """A clean shock from an analysis made after both pads went on during compressions; before the shock no pause in
    compressions over 10 s (from the start, between compressions, before the shock); the first compression after it
    within 3 s and compressions kept up 5 s; both pads on their sites and the left gripper off the manikin; both pads
    still for 3 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate read for
    this step)."""
    pad_a, pad_b = objects["pad_a"], objects["pad_b"]
    arms = rigstate.arms()
    result = _TRACKER.update(pad_a, pad_b, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "pad_a_pos": pad_a.pos, "pad_b_pos": pad_b.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if at any moment a press of the shock button began while a gripper was at the patient, a gripper was at the
    patient while the button was pressed on the charged AED, a pad rested on the compression mark, or a pad fell off
    the table. Else 1.0 a qualified shock (from an analysis after both pads went on during compressions) with all
    three timings met, both pads on their sites and the left gripper off the manikin; 0.75 a qualified shock; 0.5 a
    qualified analysis; 0.25 both pads on their sites at once at some moment; 0 otherwise."""
    return _TRACKER.result(objects["pad_a"], objects["pad_b"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PAD_GRIP = -(L.PAD_SIZE[0] - 0.008) / 2  # 32 mm across a 40 mm pad (the pads are square: either axis works)
FIST = -0.008  # fully closed, 16 mm commanded, nothing held: the compressing hand and the button-pressing hand
# The left arm uses yaw -pi/2 and the right +pi/2, the signs that reach the midline. The pads are square and every
# finger clearance below holds whichever horizontal axis the fingers close along.
YAW_L = -math.pi / 2
YAW_R = math.pi / 2
READY_R = (0.323, -0.305, L.TOP + 0.26)  # yam_arena's ready pose for the right arm
TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.004  # the pads are gripped with the finger tips 4 mm above their bottom (20 of their 24 mm held)
PICK_Z = L.TOP + GRIP_UP + TIPS
APPROACH = 0.030  # each final descent onto a pad starts this far above the grip height and takes 1.5 s
CARRY_Z = L.TORSO_TOP + 0.030 + GRIP_UP + TIPS  # a carried pad's bottom 30 mm above the chest and the head
PLACE_Z = L.MARK_TOP + 0.002 + GRIP_UP + TIPS  # let go with the pad's bottom 2 mm above its site
ENTRY_Z = PLACE_Z + 0.020  # the last 20 mm are lowered slowly
CLEAR_WAY = (0.300, 0.150, L.TOP + 0.140)  # leaving the apex site: up and toward the head, tips 77 mm over the chest
BUTTON_HOVER_Z = L.TOP + 0.140  # tips 76 mm above the button: the closed hand waits here
PRESS_ENTRY_Z = L.BUTTON_TOP + 0.020 + TIPS
PRESS_Z = L.BUTTON_TOP + 0.002 + TIPS  # the tips 2 mm above the button's top ('on the button' is within 6 mm)
# Compressions: the right tips go between 40 mm and 8 mm above the compression mark: 0.3 s down, 0.4 s held, 0.3 s
# up, 0.3 s held (a 1.3 s cycle, 46 a minute). 'A compression' needs 30 mm and 15 mm, so tracking that shrinks the
# 32 mm stroke by half still counts; the holds let a lagging arm reach each end (offline, every stroke still counts
# with a 0.3 s first-order lag); the 8 mm bottom keeps the closed fingers off the chest.
STROKE_TOP = L.MARK_TOP + 0.040 + TIPS
STROKE_BOTTOM = L.MARK_TOP + 0.008 + TIPS
STAND_CLEAR = L.MARK_TOP + 0.080 + TIPS  # 'stand clear': tips 80 mm above the mark, 30 mm over the clear line
STROKE_MOVE_S, BOTTOM_HOLD_S, TOP_HOLD_S = 0.3, 0.4, 0.3
CYCLE_S = 2 * STROKE_MOVE_S + BOTTOM_HOLD_S + TOP_HOLD_S
HALF_STROKE_S = 0.5  # coming down onto the chest from 'stand clear', and lifting off it
PRESS_MARGIN_S = 1.0  # the press begins this long after the plan's moment of the AED charging
RESUME_MARGIN_S = 0.6  # the right hand starts back down this long after the plan's moment of the shock
POST_STROKES = 9  # strokes after the shock: 11.7 s, for the 5 s needed and the 3 s success hold


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _pick(x, y, name) -> list[Waypoint]:
    return [
        Waypoint(1.5, (x, y, CARRY_Z), OPEN, f"over {name}", YAW_L),
        Waypoint(1.0, (x, y, PICK_Z + APPROACH), OPEN, "down to just above it", YAW_L),
        Waypoint(1.5, (x, y, PICK_Z), OPEN, "slowly down beside it", YAW_L),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (x, y, PICK_Z), PAD_GRIP, f"grip {name}", YAW_L),
        Waypoint(1.0, (x, y, CARRY_Z), PAD_GRIP, "lift", YAW_L),
    ]


def _place(x, y, name, carry_s) -> list[Waypoint]:
    return [
        Waypoint(carry_s, (x, y, CARRY_Z), PAD_GRIP, f"carry it over the {name} site", YAW_L),
        Waypoint(0.8, (x, y, ENTRY_Z), PAD_GRIP, "down to 2 cm above the site", YAW_L),
        Waypoint(0.5, (x, y, ENTRY_Z), PAD_GRIP, "settle", YAW_L),
        Waypoint(1.5, (x, y, PLACE_Z), PAD_GRIP, "lower the last 2 cm slowly", YAW_L),
        Waypoint(0.5, (x, y, PLACE_Z), PAD_GRIP, "settle", YAW_L),
        Waypoint(1.5, (x, y, PLACE_Z), OPEN, f"let go: the pad is on the {name} site", YAW_L),
        Waypoint(1.0, (x, y, CARRY_Z), OPEN, "up", YAW_L),
    ]


def _strokes(n, what) -> list[Waypoint]:
    out = []
    at = (L.CPR_X, L.CPR_Y)
    for k in range(n):
        out += [Waypoint(STROKE_MOVE_S, (*at, STROKE_BOTTOM), FIST, f"{what} {k + 1}", YAW_R),
                Waypoint(BOTTOM_HOLD_S, (*at, STROKE_BOTTOM), FIST, "hold", YAW_R),
                Waypoint(STROKE_MOVE_S, (*at, STROKE_TOP), FIST, "release", YAW_R),
                Waypoint(TOP_HOLD_S, (*at, STROKE_TOP), FIST, "hold", YAW_R)]
    return out


def _left_pads() -> list[Waypoint]:
    """Left arm, first part: pad A onto the sternal site, pad B onto the apex site, then up and away toward the
    head, to the AED, and close the hand over the button."""
    return [
        *_pick(L.PAD_A_X, L.PAD_A_Y, "pad A"),
        *_place(L.STERNAL_X, L.STERNAL_Y, "sternal", 1.5),
        Waypoint(1.0, (L.STERNAL_X, 0.250, CARRY_Z), OPEN, "toward the head, 7.5 cm from the right hand in x", YAW_L),
        *_pick(L.PAD_B_X, L.PAD_B_Y, "pad B"),
        *_place(L.APEX_X, L.APEX_Y, "apex", 2.0),
        Waypoint(1.0, CLEAR_WAY, OPEN, "up and away from the patient", YAW_L),
        Waypoint(1.5, (L.AED_X, L.AED_Y, BUTTON_HOVER_Z), OPEN, "over the shock button", YAW_L),
        Waypoint(0.5, (L.AED_X, L.AED_Y, BUTTON_HOVER_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (L.AED_X, L.AED_Y, BUTTON_HOVER_Z), FIST, "close the hand, ready to shock", YAW_L),
    ]


LEFT_READY_S = _span(_left_pads())  # the left hand is closed over the button, clear of the patient


def _crossing(z0, z1, level) -> float:
    """Fraction of a straight move from height z0 to z1 at which it passes ``level``."""
    return (level - z0) / (z1 - z0)


def _left_clear_s() -> float:
    """The plan's moment the left fingers leave the clear zone: rising past 50 mm over the chest after the apex
    pad (the gripper is open there, but straight above the chest only the height counts)."""
    wps = _left_pads()
    i = next(i for i, w in enumerate(wps) if w.label == "up and away from the patient")
    return _span(wps[:i]) + wps[i].duration * _crossing(CARRY_Z, CLEAR_WAY[2], L.TORSO_TOP + L.CLEAR_Z + TIPS)


def _right_until_clear() -> tuple[list[Waypoint], float]:
    """Right arm, first part: close the hand, come over the compression mark and compress until the left hand is
    ready at the button, then lift to stand clear. Also returns the plan's moment the right fingers are clear."""
    wps = [
        Waypoint(0.5, READY_R, OPEN, "settle", YAW_R),
        Waypoint(1.5, READY_R, FIST, "close the hand", YAW_R),
        Waypoint(1.5, (L.CPR_X, L.CPR_Y, STAND_CLEAR), FIST, "over the compression mark", YAW_R),
        Waypoint(HALF_STROKE_S, (L.CPR_X, L.CPR_Y, STROKE_TOP), FIST, "hands on the chest", YAW_R),
        Waypoint(TOP_HOLD_S, (L.CPR_X, L.CPR_Y, STROKE_TOP), FIST, "hold", YAW_R),
    ]
    n = math.ceil((max(LEFT_READY_S, _left_clear_s()) - _span(wps)) / CYCLE_S - 1e-9)
    wps += _strokes(n, "compression")
    lift = Waypoint(HALF_STROKE_S, (L.CPR_X, L.CPR_Y, STAND_CLEAR), FIST, "stand clear for the analysis", YAW_R)
    clear_s = _span(wps) + lift.duration * _crossing(STROKE_TOP, STAND_CLEAR, L.TORSO_TOP + L.CLEAR_Z + TIPS)
    return wps + [lift], clear_s


CLEAR_S = max(_left_clear_s(), _right_until_clear()[1])  # the plan's moment both hands are clear
CHARGED_S = CLEAR_S + L.ANALYSIS_S  # the plan's moment the AED's analysis completes
PRESS_S = CHARGED_S + PRESS_MARGIN_S  # the plan's moment the left tips come within 6 mm of the button
# The press entry: 20 mm above the button to 2 mm in 0.8 s; the tips pass 6 mm above it 14/18 of the way down.
_PRESS_IN = 0.8 * _crossing(PRESS_ENTRY_Z, PRESS_Z, L.BUTTON_TOP + L.PRESS_Z + TIPS)
SHOCK_S = PRESS_S + L.PRESS_S  # the plan's moment the shock is delivered


def _left() -> list[Waypoint]:
    """Left arm (AED and pads): both pads on while the right arm compresses, then to the AED, wait clear while it
    analyses, press the shock button, lift the hand and stay clear."""
    pads = _left_pads()
    down_s = 0.8
    wait = PRESS_S - _PRESS_IN - down_s - _span(pads)
    hx, hy = L.AED_X, L.AED_Y
    return pads + [
        Waypoint(wait, (hx, hy, BUTTON_HOVER_Z), FIST, "wait, clear, while the AED analyses", YAW_L),
        Waypoint(down_s, (hx, hy, PRESS_ENTRY_Z), FIST, "down to 2 cm above the button", YAW_L),
        Waypoint(0.8, (hx, hy, PRESS_Z), FIST, "press the shock button", YAW_L),
        Waypoint(1.0, (hx, hy, PRESS_Z), FIST, "hold it: the shock", YAW_L),
        Waypoint(0.8, (hx, hy, BUTTON_HOVER_Z), FIST, "lift the hand off the button", YAW_L),
    ]


def _right() -> list[Waypoint]:
    """Right arm (compressions): compress from the start until the left hand is ready, stand clear for the analysis
    and the shock, then straight back to compressions."""
    wps = _right_until_clear()[0]
    wait = SHOCK_S + RESUME_MARGIN_S - _span(wps)
    wps.append(Waypoint(wait, (L.CPR_X, L.CPR_Y, STAND_CLEAR), FIST, "stay clear until the shock", YAW_R))
    wps.append(Waypoint(HALF_STROKE_S, (L.CPR_X, L.CPR_Y, STROKE_TOP), FIST, "hands back on the chest", YAW_R))
    wps.append(Waypoint(TOP_HOLD_S, (L.CPR_X, L.CPR_Y, STROKE_TOP), FIST, "hold", YAW_R))
    return wps + _strokes(POST_STROKES, "post-shock compression")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
