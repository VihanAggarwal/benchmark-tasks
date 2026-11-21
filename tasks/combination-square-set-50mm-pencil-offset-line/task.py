"""On the station table, real size: a 1-by-6 board (3/4 x 5-1/2 x 24 in) lies along the table. A STANLEY 12 in combination
square, its head already locked on the blade at a 49 mm gauge setting, lies on the table 30 mm off the board's near edge,
its blade over the board. A flat carpenter's pencil stands tip down in a wooden pencil block beside the board.

Goal: a line 50 mm in from the near edge, parallel to it, at least 90 mm long: seat the head's face on the edge, press
the pencil's lead on the board against the blade's end, and slide square and pencil along together. Then the pencil back in
its block and both hands away. (A pencil line cannot be drawn in the simulator: the scorer reads the PATH of the lead's
tip while the head is held against the edge; in a real trial the pencil's graphite draws that path.)

The parts are rigid mechkit bodies (mechkit.py, next to this file). The square slides and turns by ordinary contact
friction, no rule. The one declared rule is ink: each step the lead's tip touches the board's top face under a measured
contact force is a point of the path (scoring.py).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import contact  # noqa: E402
import offline_model  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import BLOCK, BOARD, PENCIL, SQUARE  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()
contact.install(PENCIL, BOARD)  # contact reports on both bodies, read by a PhysX rigid-contact view (sim)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Draw a line 50 mm in from the board's near edge and parallel to it, at least 90 mm long, with the "
               "combination square and the pencil: seat the square's head on the edge, hold the pencil's lead on the board "
               "against the end of the blade, and slide both along together. Then stand the pencil back in its block and let "
               "go of everything.")

# -- the parts --------------------------------------------------------------------------------------------------------
BOARD_PART = mk.assembly(BOARD, origin=L.BOARD_ORIGIN, shapes=P.board_shapes(), mass=L.BOARD_MASS, color=L.PINE,
                         friction=L.BOARD_FRICTION)
SQUARE_PART = mk.assembly(SQUARE, origin=L.SQ_START, shapes=P.square_shapes(), mass=L.SQ_MASS, color=L.HEAD_COLOR,
                          friction=L.SQ_FRICTION)
PENCIL_PART = mk.assembly(PENCIL, origin=L.PN_START, shapes=P.pencil_shapes(), mass=L.PN_MASS, color=L.PN_RED,
                          friction=L.PN_FRICTION, tight_fit=True)
BLOCK_PART = mk.assembly(BLOCK, origin=L.BLOCK_ORIGIN, shapes=P.block_shapes(), mass=None, color=L.BLOCK_WOOD,
                         friction=L.PN_FRICTION, tight_fit=True)  # the pencil block, fixed
OBJECTS = [BOARD_PART, SQUARE_PART, PENCIL_PART, BLOCK_PART]

_OFFLINE = offline_model.Model(BOARD_PART, SQUARE_PART, PENCIL_PART, BLOCK_PART)


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(SQUARE), "source", "sim") != "sim"


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
        except OSError:
            pass


def success(objects) -> bool:
    """A good line, the pencil back in its block, hands off and still for the last 1 s, no latch. This runs first every
    control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():
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
        force = None
    else:
        force = contact.force()
    result = _TRACKER.update(parts, arms, force=force, geometric=offline)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update({f"{n}_origin": parts[n].origin for n in S.PARTS})
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
    if not offline and contact.error():
        row["contact_error"] = contact.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a latch fired or the square was never seated; 0.25 seated against the edge for 0.5 s but no good line; 0.5 a
    good line, the pencil not back in its block; 0.75 good and stowed, not let go and still for the last 1 s; 1.0 all
    of it. The latest scored step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout. Both grips are straight down: the right closing along world
# x, the left along world y. The arms work crossed (the right hand holds the pencil at the blade's end; the left hand
# grips the blade's tail behind the head, about 160 mm nearer the robots on the same y). Points are given for the pad
# tips (left: beside the blade's tail; right: the pencil's lead tip); the TCP is 13 mm above the pad tips.
PAD = 0.013
F_OPEN = -0.025  # fingers 50 mm apart
F_SHUT = 0.0
X_SEAT = L.NEAR_X - L.SEAT_GAP
PUSH_HOME = X_SEAT - L.SQ_START[0] + L.PRESS_X_BIAS  # how far the left hand pushes the head toward the edge


def _tcp(x, y, z_tips):
    return (float(x), float(y), L.TOP + float(z_tips) + PAD)


def _pen(x, y, z_tip):
    """The right TCP that puts the held pencil's lead tip at (x, y, z_tip), table frame (the body's middle is the grip)."""
    return (float(x) + L.PN_CENTER_X, float(y), L.TOP + float(z_tip) + L.PN_GRIP_Z)


def _wp(d, pos, finger, label, yaw):
    return Waypoint(float(d), pos, finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _hold(plan, until, label):
    last = plan[-1]
    gap = until - _duration(plan)
    if gap > 1e-9:
        plan.append(Waypoint(gap, last.pos, last.finger, label, last.yaw))


LANE_UP = 0.025  # the lead's height over the board in the lane
T_SEATED = 20.0  # by now the left hand has pushed the head home against the edge
T_RUN = 26.5  # both hands start along the board together


def run_time(run_end: float = L.RUN_B) -> float:
    return (run_end - L.RUN_A) / L.DRAW_SPEED


def stroke_steps(y0: float, y1: float) -> list:
    """(duration, y) of the stroke's waypoints: DRAW_SEG apart at DRAW_SPEED. Each waypoint is smoothstepped, so one
    long waypoint would run the middle of the stroke at 1.5 times the mean speed (GPU run 3: 15 mm/s for a 10 mm/s
    stroke, too few steps in each 2 mm stretch); short equal ones give every stretch of the line the same time. Both
    hands use the same list, so square and pencil travel in step."""
    n = max(1, int(round(abs(y1 - y0) / L.DRAW_SEG)))
    dt = run_time(L.RUN_A + (y1 - y0)) / n
    return [(dt, y0 + (y1 - y0) * (k + 1) / n) for k in range(n)]


def blade_end_x(push: float = PUSH_HOME) -> float:
    """World x of the blade's end once the head has been pushed ``push`` toward the edge (it stops at the edge)."""
    x_face = min(L.SQ_START[0] + push, X_SEAT)
    return x_face + L.SQ_EXT


def right_moves(run_end: float = L.RUN_B, block_dy: float = 0.0, push: float = PUSH_HOME) -> list:
    """Over the pencil, grip its body, lift it out of its block, over the board beyond the blade's end, lead down onto
    the board just beyond the blade's end, in against the end, along the board with the square to the far end, lead up,
    out, back over the block, down to just over its top, slowly past the walls' top into the slot, on down to the floor,
    let go, wait, straight up along the pencil, aside. ``run_end``, ``block_dy`` and ``push`` make the
    counterfactuals."""
    tip_bx = L.BL_X - L.PN_CENTER_X  # the lead tip's x when the pencil stands in its slot
    by = L.BL_Y
    yaw = L.YAW_R
    z_draw = L.BOARD_H - L.LEAD_DOWN
    x_end = blade_end_x(push)
    x_app = x_end - L.PN_X0 + L.PEN_APP_GAP  # the pencil's face 1.5 mm beyond the blade's end (it comes down there)
    x_in = x_end - L.PN_X0 - L.PEN_PRESS  # the face 1.0 mm into it
    z_hover = L.BOARD_H + L.PEN_HOVER  # the lead's tip just over the board, under the blade's top: no ink yet
    sy = L.BL_Y + block_dy + L.STOW_DY  # where the pencil goes back in (see layout.STOW_DY) ...
    sx = tip_bx + L.STOW_DX  # ... and STOW_DX (layout: why every GPU run so far missed the slot in x), added
    #                          after the hover (the hover itself, as in the GPU runs, has joint4 5 mrad off its limit)
    z_hover_bl = L.BL_FLOOR + 0.055  # the tip 10 mm over the block's top
    z_over_bl = L.BL_H + L.STOW_OVER  # the tip 3 mm over it
    z_entry = z_over_bl - L.STOW_ENTRY  # the tip 12 mm into the slot
    y0 = L.RUN_A
    y1 = run_end
    fl = L.BL_FLOOR
    x_side = x_app + 0.045  # a lane beyond the blade's end, clear of the head hand
    x_lane = x_app + 0.085  # the lane the pencil goes back by: well clear of the left arm
    y_lane = -0.190  # the lane's y: well on the right hand's own side, clear of the left arm
    plan = [
        _wp(2.5, _pen(tip_bx, by - 0.080, fl), F_OPEN, "beside the pencil, the pads either side of its body's height", yaw),
        _wp(1.5, _pen(tip_bx, by, fl), F_OPEN, "in along y until the pencil is between the fingers", yaw),
        _wp(0.5, _pen(tip_bx, by, fl), F_OPEN, "settle", yaw),
        _wp(1.5, _pen(tip_bx, by, fl), L.PN_PINCH, "grip the body", yaw),
        _wp(0.5, _pen(tip_bx, by, fl), L.PN_PINCH, "settle the grip", yaw),
        _wp(2.0, _pen(tip_bx, by, fl + 0.055), L.PN_PINCH, "lift the pencil straight up out of its block", yaw),
        _wp(3.0, _pen(x_side, y_lane, L.BOARD_H + LANE_UP), L.PN_PINCH, "over the board's far side to the lane beyond the blade's end", yaw),
    ]
    _hold(plan, T_SEATED + 0.5, "wait for the head to be pushed home")
    plan += [
        _wp(3.0, _pen(x_app, y0, L.BOARD_H + 0.010), L.PN_PINCH, "lead to just over the board, beyond the blade's end", yaw),
        _wp(1.0, _pen(x_app, y0, z_hover), L.PN_PINCH, "lead down to 1 mm over the board, beyond the blade's end", yaw),
        _wp(1.0, _pen(x_in, y0, z_hover), L.PN_PINCH, "in against the blade's end, the lead still off the board", yaw),
        _wp(0.8, _pen(x_in, y0, z_draw), L.PN_PINCH, "lead down onto the board, the face on the blade's end", yaw),
    ]
    _hold(plan, T_RUN, "settle against the blade's end")
    plan += [
    ]
    plan += [_wp(dt, _pen(x_in, y, z_draw), L.PN_PINCH,
                 "draw: along the board with the square, pencil on the blade's end, 3.4 mm/s", yaw)
             for dt, y in stroke_steps(y0, y1)]
    plan += [
        _wp(0.4, _pen(x_in, y1, z_draw), L.PN_PINCH, "stop", yaw),
        _wp(1.0, _pen(x_in, y1, L.BOARD_H + 0.015), L.PN_PINCH, "lead straight up off the board", yaw),
        _wp(1.0, _pen(x_side, y1, L.BOARD_H + 0.015), L.PN_PINCH, "out beyond the blade's end", yaw),
        _wp(2.5, _pen(x_lane, y_lane, L.BOARD_H + LANE_UP), L.PN_PINCH, "along the lane to the right hand's own side", yaw),
        _wp(3.0, _pen(tip_bx, sy, z_hover_bl), L.PN_PINCH, "over the block", yaw),
        _wp(L.STOW_SETTLE_S, _pen(tip_bx, sy, z_hover_bl), L.PN_PINCH, "settle over the block", yaw),
        _wp(L.STOW_OVER_S, _pen(sx, sy, z_over_bl), L.PN_PINCH, "down to 3 mm over the block's top, 0.7 mm to +x", yaw),
        _wp(L.STOW_WAIT_S, _pen(sx, sy, z_over_bl), L.PN_PINCH, "wait: the hand's lag dies out", yaw),
        _wp(L.STOW_ENTRY_S, _pen(sx, sy, z_entry), L.PN_PINCH,
            "slowly into the slot, past the walls' top, 12 mm in", yaw),
        _wp(L.STOW_DOWN_S, _pen(sx, sy, fl + 0.0005), L.PN_PINCH, "on down inside the slot, tip on the floor", yaw),
        _wp(L.STOW_HOLD_S, _pen(sx, sy, fl + 0.0005), L.PN_PINCH, "settle, still held", yaw),
        _wp(1.5, _pen(sx, sy, fl + 0.0005), F_OPEN, "let go", yaw),
        _wp(L.STOW_OPEN_WAIT_S, _pen(sx, sy, fl + 0.0005), F_OPEN, "wait, the pencil standing free in the slot", yaw),
        _wp(L.STOW_LIFT_S, _pen(sx, sy, fl + 0.0005 + L.STOW_LIFT), F_OPEN,
            "straight up along the pencil, the open fingers either side of it", yaw),
        _wp(2.0, _pen(sx, sy - 0.080, fl + 0.055), F_OPEN, "aside along -y and a little up, well clear", yaw),
    ]
    return plan


def draw_times(run_end: float = L.RUN_B) -> tuple[float, float]:
    """When the stroke starts and when the lead is up off the board (s from the start)."""
    t = 0.0
    start = end = None
    for w in right_moves(run_end):
        t += w.duration
        if w.label.startswith("settle against"):
            start = t
        if w.label.startswith("lead straight up"):
            end = t
    return start, end


def left_moves(run_end: float = L.RUN_B, push: float = PUSH_HOME) -> list:
    """Fingers open across the blade's tail behind the head, down until the pads are either side of it, grip it by its
    two edges, push the square along the table until the head's face is on the edge (and a little past, held there by
    the arm), hold it there while the pencil goes in against the blade's end, slide it along the edge with the pencil
    at the same pace, hold it until the pencil is up and clear, let go, up, aside."""
    yaw = L.YAW_GRIP
    gx0 = L.SQ_START[0] + L.GRIP_LOCAL[0]
    gy0 = L.SQ_START[1] + L.GRIP_LOCAL[1]
    z_over = L.SQ_BLADE_Z + L.SQ_BLADE_T + 0.040  # the pad tips 40 mm over the blade
    z_grip = L.GRIP_TIPS_Z
    run = run_end - L.RUN_A
    plan = [_wp(4.0, (0.200, 0.200, L.TOP + 0.100), F_OPEN, "aside on its own side while the pencil is fetched", yaw)]
    _hold(plan, 11.5, "wait")
    plan += [
        _wp(2.5, _tcp(gx0, gy0, z_over), F_OPEN, "fingers open across the blade's tail, behind the head", yaw),
        _wp(1.5, _tcp(gx0, gy0, z_grip), F_OPEN, "down, the pads either side of the blade", yaw),
        _wp(1.5, _tcp(gx0, gy0, z_grip), L.BLADE_PINCH, "grip the blade by its edges", yaw),
        _wp(0.5, _tcp(gx0, gy0, z_grip), L.BLADE_PINCH, "settle the grip", yaw),
        _wp(2.5, _tcp(gx0 + push, gy0, z_grip), L.BLADE_PINCH,
            "push the square along the table until the head's face is on the edge", yaw),
    ]
    for k in range(2):  # sliding friction lets the arm close its x error (GPU run 2 stalled 6.4 mm short)
        plan += [
            _wp(0.5, _tcp(gx0 + push, gy0 + L.SEAT_WIGGLE, z_grip), L.BLADE_PINCH,
                "work the face onto the edge: along +y, still pushed toward the edge", yaw),
            _wp(0.5, _tcp(gx0 + push, gy0, z_grip), L.BLADE_PINCH, "work the face onto the edge: back", yaw),
        ]
    _hold(plan, T_RUN, "hold it on the edge while the pencil goes in against the blade's end")
    plan += [_wp(dt, _tcp(gx0 + push, gy0 + y - L.RUN_A, z_grip), L.BLADE_PINCH,
                 "slide it along the edge with the pencil, still held toward the edge, 3.4 mm/s", yaw)
             for dt, y in stroke_steps(L.RUN_A, run_end)]
    _hold(plan, draw_times(run_end)[1] + 7.5, "hold it until the pencil is back over its block")
    plan += [
        _wp(1.0, _tcp(gx0 + push, gy0 + run, z_grip), F_OPEN, "let go of the blade", yaw),
        _wp(1.0, _tcp(gx0 + push, gy0 + run, z_over), F_OPEN, "up off the blade", yaw),
        _wp(1.5, _tcp(gx0 + push, gy0 + run, z_over + 0.070), F_OPEN, "straight up, well clear of the square", yaw),
    ]
    return plan


# Wrong ways (counterfactuals; the offline model predicts them):
#   short    both hands stop 45 mm short of the far end: the line is under 90 mm: 0.25
#   unseated the head pushed only 15 mm, left 15 mm off the edge: never seated, the line not guided: 0
#   nostow   the pencil put back down 50 mm beside its block, not in it: 0.5
VARIANTS = ("short", "unseated", "nostow")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "short":
        end = L.RUN_B - 0.045
        return {"left": left_moves(run_end=end), "right": right_moves(run_end=end)}
    if variant == "unseated":
        return {"left": left_moves(push=0.015), "right": right_moves(push=0.015)}
    if variant == "nostow":
        return {"left": left_moves(), "right": right_moves(block_dy=0.050)}
    return {"left": left_moves(), "right": right_moves()}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan. DEMO_IDLE_ARM=left|right: that arm's plan is dropped.
    DEMO_VARIANT: see VARIANTS."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
