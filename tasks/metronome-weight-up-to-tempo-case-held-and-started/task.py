"""On the station table, real size: a Wittner Taktell Piccolo clockwork metronome (155 x 70 x 45 mm, 171 g), standing
loose, its front with the pendulum and the tempo scale toward the right arm (-y). It is wound; its pendulum hangs still and its
sliding weight sits at the 160 mark, where the last practice left it.

Goal: set it to 100 and start it: slide the weight up the rod to the 100 mark while the case is held down, let go,
then swing the pendulum out and let it tick, both hands away.

The metronome is one loose mechkit articulation (mechkit.py, next to this file): the case is the base; the pendulum
(rod and its counterweight inside the case) turns on a hinge about x; the weight rides a slider along the rod, held by
its spring clip: a stiff spring toward a target that one rule (clip.py, the mechkit on_step callback ``_clip_rule``
below) keeps still until the weight is pushed or pulled along the rod with 3 N and otherwise moves so the spring pulls
back with 3 N (PhysX joint friction let the weight creep with no force on it on the GPU, 19 Nov). The case and its
parts weigh 1.68 N, so a weight dragged up without the case held lifts the case off the table. The tempo is the
pendulum's own swing rate (pendulum.py gives it from the same masses and shapes); the scorer times the swings. The
rule knows no mark and no goal; there is no flag.

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

import clip  # noqa: E402
import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the metronome is built as an articulation)
import offline_model  # noqa: E402
import parts as P  # noqa: E402
import pendulum as PEN  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import METRONOME  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("The metronome is still at 160 from the last piece. Set it to 100 for the next piece and start it "
               "ticking on its own.")

METRONOME_PART = mk.assembly(METRONOME, origin=L.M_ORIGIN, shapes=P.case_shapes(), mass=L.BASE_MASS,
                             joints=P.joints(), yaw=L.M_YAW, color=L.BLACK, friction=L.CASE_FRICTION)
OBJECTS = [METRONOME_PART]

_OFFLINE = offline_model.MetronomeModel(METRONOME_PART)


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(METRONOME), "source", "sim") != "sim"


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
    CLIP.reset()
    mk.set_targets(METRONOME_PART, "weight", 0.0)


# -- the one rule: the weight's spring clip (clip.py) ----------------------------------------------------------------
CLIP = clip.Clip()
RULE = {"skipped": 0, "slipping": False}


@mk.on_step
def _clip_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics): the clip's target
    from the weight's place on the rod. An unreadable step keeps the last target."""
    try:
        q = float(states[METRONOME].joints["weight"])
    except (AttributeError, KeyError, TypeError, ValueError):
        q = math.nan
    if not math.isfinite(q):
        RULE["skipped"] += 1
    before = CLIP.slips
    t = CLIP.update(q)
    slipping = CLIP.slips != before
    if slipping != RULE["slipping"]:
        print(f"[metronome] clip {'slides' if slipping else 'holds'} at episode {mk.episode()} step {mk.step()}: "
              f"weight {(PEN.R_START + q) * 1000:.2f} mm over the pivot, target {(PEN.R_START + t) * 1000:.2f} mm",
              file=sys.stderr, flush=True)
        RULE["slipping"] = slipping
    mk.set_targets(METRONOME_PART, "weight", t)


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only
            pass


def success(objects) -> bool:
    """Ticking at 96 to 104 on its own, standing, both hands clear. Runs first every control step, so it also feeds the
    tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    parts = {METRONOME: objects.get(METRONOME)}
    if no_simulator(parts):
        _OFFLINE.step(arms)
        parts = {METRONOME: _OFFLINE.state()}
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "score": result.score, "success": result.success, **result.details}
    try:
        row["origin"] = parts[METRONOME].origin
        row["joints"] = parts[METRONOME].joints
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
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 ticking at 96 to 104 per minute with both hands 30 mm clear; 0.75 at 90 to 110; 0.5 the weight set at the
    100 mark (within 2.5 mm) but not ticking on its own; 0.25 ticking at another tempo; 0 otherwise. Ticks count only
    while the case stands on the table, hands clear. The latest step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout. Points are given in the case's base frame (x toward its
# back, y across its front, z up from the table; layout.py) and turned into world TCP targets by _w. Both grips close
# along world x, i.e. across the case's front (right yaw +pi/2, left yaw -pi/2: the same line, each within 90 deg of
# its arm's bearing), rolled about world x (below). The pads' flat faces run from 9 mm above the TCP to their tips 13 mm below it.
import tiltgrip  # noqa: E402  (rolled grips for the demo: wraps yam_arena.demo.grip_rotation)

# Rolled about world x (tiltgrip.py), so the two wrists lean apart: the right's tips toward +y (its wrist toward -y),
# the left's toward -y. The pads stay normal to world x, so both still pinch across x. Straight down, the two grips
# 55 mm apart in y put the left wrist into the right gripper (arm capsules 57 mm into each other); rolled 0.45 rad
# each, the capsules clear by 32 mm or more at every hold.
ROLL_R, ROLL_L = 0.45, -0.45
YAW_R = tiltgrip.tilted(math.pi / 2, ROLL_R)
YAW_L = tiltgrip.tilted(-math.pi / 2, ROLL_L)
_C, _S = math.cos(L.M_YAW), math.sin(L.M_YAW)


def world(p) -> tuple:
    """A base-frame point (x, y, z above the table) as a world TCP target."""
    return (L.M_X + _C * p[0] - _S * p[1], L.M_Y + _S * p[0] + _C * p[1], L.TOP + p[2])


def fingers(gap: float) -> float:
    return -gap / 2


ROD_X = L.ROD_X
W_X = ROD_X + L.W_X0 + L.W_SIZE[0] / 2
GRIP_X = W_X - 0.003  # 3 mm forward of the weight's middle: the pads' back edges 2.5 mm off the case's front
Z_START = L.PIVOT_Z + PEN.R_START  # the weight's centre at 160
Z_GOAL = L.PIVOT_Z + PEN.R_GOAL  # at 100
W_OPEN = fingers(0.036)
W_GRIP = fingers(L.W_SIZE[1] - 0.008)
HOLD_X = 0.022  # over the case's back half (base x), toward the left arm
HOLD_Z = 0.036  # the rolled pads' lowest corner 18 mm over the table when the pinch closes
# Then the pinch is pressed 10 mm down (commanded; the pads' lowest corner 8 mm over the table if they never slip), so
# the arm's give pushes the case onto the table with 2 to 5 N. GPU 19 Nov: without it the case rose 6.5 mm and tilted
# 1.6 deg with the dragged weight (the left arm gave 5.4 mm under the 1.4 N net pull) and took 6 mm of the drag.
PRESS_DZ = 0.010
C_OPEN = fingers(0.084)
C_GRIP = fingers(2 * L.TIERS[0][2] - 0.008)
CLOSED = 0.0
# The start: the closed pads push the WEIGHT's side (world +x face) out toward the arms, from in front of the case.
# GPU 19 Nov, second run: the old push (rolled pads beside the rod's top, 7.6 mm over the weight) came down on the
# case's top instead. The 0.45 rad roll puts the pad tips 5.7 mm toward the case and their 27 mm width leans up
# toward it, and the case had slid 11.5 mm toward the right arm during the drag (the left's press, see DRAG_GIVE):
# the closed fingers stalled 18 mm over their command on the upper tier's front edge, tilted the case 2.5 deg, then
# dragged its top along and tipped it 15 deg; the rod itself was never touched (it rocked 3 deg once the case fell
# back), so no timed ticks. Now the push is made straight down (the left hand is parked by then, so no roll is
# needed), the pads' edge toward the case 3 mm short of where the case's front (with its scale strip and marks)
# would be after a slide of CASE_SHIFT toward the right arm, and so 16 mm off it if it has not slid; the pads then
# cover 11 mm or more of the weight's 20 mm depth (the weight stands 16 mm proud of the rod) for any slide from
# 1 mm away to 13 mm toward the right arm, and never the rod (3 mm in front of it, unslid). Their tips sit 6 mm over
# the middle tier's top (110 mm), on the weight's side face (114 to 128 mm set). Let go by sliding the pads off the
# weight's front toward the right arm (no lift past the rod), so it swings from where it was held.
YAW_PUSH = math.pi / 2  # straight down, the pads closing along world x
PAD_HALF = 0.0135  # the closed pads' half width, along base x for this grip
CASE_FRONT = L.TIERS[2][0] - P.SCALE_X - 0.0006  # base x of the upper tier's front with its strip and marks: -17.6 mm
CASE_SHIFT = 0.013  # the case may stand this much nearer the right arm by the start (GPU 19 Nov: 11.5 mm)
PUSH_X = CASE_FRONT - CASE_SHIFT - 0.003 - PAD_HALF  # -47.1 mm: the pads' edge toward the case at -33.6 mm
PUSH_TIP = L.TIERS[1][4] + 0.006  # pad tips 116 mm up
PUSH_Z = PUSH_TIP + 0.013
FACE = 0.008  # the closed pads' outer face from the TCP, along the closing axis (base y)
W_SIDE = L.W_SIZE[1] / 2  # the weight's side face from the rod's line (base y)
PUSH_Y0 = -(W_SIDE + 0.008 + FACE)  # the pads' face 8 mm off the weight's side, on its base -y side (world +x)
PUSH_AMP = math.radians(27.0)  # the swing it is held out at and let go from (20 to 30 deg wanted)


def push_face(amp: float, tip: float = PUSH_TIP) -> float:
    """Base y of the pads' face (from the rod's rest line) that holds the pendulum out at ``amp`` (rad) toward base
    +y: the weight's base -y edge meets the pads' tip line (the pads are vertical, the weight leans away from them)."""
    h = tip - L.PIVOT_Z
    s = (h - W_SIDE * math.sin(amp)) / math.cos(amp)  # where the tip line meets the edge, along the rod
    return s * math.sin(amp) - W_SIDE * math.cos(amp)


PUSH_Y1 = push_face(PUSH_AMP) - FACE  # the pads' face 22.1 mm past the rod's line, the TCP 14.1 mm
RETRACT_X = PUSH_X - 0.035  # slid toward the right arm until the pads' edge is 4 mm or more past the weight's front
# Straight down near the right arm, the wrist (joint4) nears its limit higher up (4 deg margin at 180 mm, 12 at
# 150, offline IK): the push's approach and its exit stay low. Neither is over the case: the approach is in front of
# it, 9 mm over the set weight's top (128 mm), and the exit in front of the weight's front.
PUSH_HIGH = 0.150  # pad tips 137 mm up
PUSH_CLEAR = 0.160  # pad tips 147 mm up, in front of everything
PARK_L = (0.200, 0.300, 0.160)  # world
PARK_R = (0.200, -0.280, 0.160)
WAIT_R = (0.180, -0.220, 0.180)
WEIGHT_TRAVEL = Z_GOAL - Z_START  # 22.7 mm: the 160 mark to the 100 mark
LIFT_DZ = WEIGHT_TRAVEL + L.DRAG_GIVE  # the hand is commanded past the mark by the arm's give under the clip's 3 N
RELAX_DZ = 0.004  # then lowered 4 mm before the fingers open: the arm's pull falls from 3 N to about 0 for an arm of
# 1 to 3.5 mm/N, inside the clip's 3 N either way (the weight stays), so the opening pads do not let a stretched arm
# jump up past the weight (the GPU hand jumped 7.5 mm as the pads opened, 19 Nov)
DRAG_S = 5.0  # slow: the clip's target is written once per 50 ms, so the pull ripples by its spring times a step's travel


def _w(d, p, finger, label, yaw):
    return Waypoint(d, world(p), finger, label, yaw)


def _park(d, xyz, finger, label, yaw):
    return Waypoint(d, (xyz[0], xyz[1], L.TOP + xyz[2]), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _hold(plan, until, label):
    last = plan[-1]
    gap = until - _duration(plan)
    if gap > 1e-9:
        plan.append(Waypoint(gap, last.pos, last.finger, label, last.yaw))


RELEASE_L = 20.0  # the left lets go once the right has let go of the weight and lifted clear


def left_moves(release_at: float = RELEASE_L) -> list:
    """Over the case's back half, down either side of its lower tier, pinch it, press it down onto the table and hold
    it there while the right hand drags the weight up, then open, up, aside."""
    plan = [
        _w(2.5, (HOLD_X, 0.0, 0.100), C_OPEN, "over the case's back half, fingers 84 mm apart", YAW_L),
        _w(1.5, (HOLD_X, 0.0, HOLD_Z), C_OPEN, "down either side of the lower tier", YAW_L),
        _w(0.5, (HOLD_X, 0.0, HOLD_Z), C_OPEN, "settle", YAW_L),
        _w(1.5, (HOLD_X, 0.0, HOLD_Z), C_GRIP, "pinch the case across its 70 mm width", YAW_L),
        _w(0.5, (HOLD_X, 0.0, HOLD_Z), C_GRIP, "settle the pinch", YAW_L),
        _w(1.0, (HOLD_X, 0.0, HOLD_Z - PRESS_DZ), C_GRIP, "press the case down onto the table", YAW_L),
    ]
    _hold(plan, release_at, "hold the case down while the weight is dragged up")
    plan += [
        _w(1.5, (HOLD_X, 0.0, HOLD_Z - PRESS_DZ), C_OPEN, "open", YAW_L),
        _w(1.5, (HOLD_X, 0.0, 0.110), C_OPEN, "straight up", YAW_L),
        _park(2.0, PARK_L, C_OPEN, "aside", YAW_L),
    ]
    return plan


def right_moves(lift: float = LIFT_DZ, start: bool = True) -> list:
    """Over the weight, down round it (the rod between the open fingers), grip it, drag it 22.7 mm up the rod to
    the 100 mark (the hand commanded 14.2 mm farther: the arm's give under the clip, the case's rise, the pads' slip
    and the clip's set, from the GPU trace), ease the pull off, open, straight up past the rod's tip; out of the left
    hand's way until it has gone; then, wrist straight down and fingers closed, down in front of the case beside the
    weight on its far side, push the weight out toward the arms, slide off its front so it swings free, aside."""
    plan = [_w(2.5, (GRIP_X, 0.0, 0.180), W_OPEN, "over the weight, fingers 36 mm apart", YAW_R)]
    _hold(plan, 6.0, "wait for the left pinch")
    plan += [
        _w(2.0, (GRIP_X, 0.0, Z_START), W_OPEN, "down round the weight, the rod between the fingers", YAW_R),
        _w(0.5, (GRIP_X, 0.0, Z_START), W_OPEN, "settle", YAW_R),
        _w(1.5, (GRIP_X, 0.0, Z_START), W_GRIP, "grip the weight across its 16 mm", YAW_R),
        _w(0.5, (GRIP_X, 0.0, Z_START), W_GRIP, "settle the grip", YAW_R),
        _w(DRAG_S, (GRIP_X, 0.0, Z_START + lift), W_GRIP, "drag the weight up the rod against its clip", YAW_R),
        _w(0.5, (GRIP_X, 0.0, Z_START + lift), W_GRIP, "settle: the arm stalls on the 100 mark", YAW_R),
        _w(1.0, (GRIP_X, 0.0, Z_START + lift - RELAX_DZ), W_GRIP, "ease the pull off (the clip holds)", YAW_R),
        _w(1.5, (GRIP_X, 0.0, Z_START + lift - RELAX_DZ), W_OPEN, "let go of the weight", YAW_R),
        _w(0.5, (GRIP_X, 0.0, Z_START + lift - RELAX_DZ), W_OPEN, "settle", YAW_R),
        _w(1.5, (GRIP_X, 0.0, 0.180), W_OPEN, "straight up, clear of the rod's tip", YAW_R),
    ]
    if not start:
        plan.append(_park(2.0, PARK_R, W_OPEN, "aside", YAW_R))
        return plan
    plan += [_park(1.5, WAIT_R, CLOSED, "out of the left hand's way, fingers closing, wrist unrolled", YAW_PUSH)]
    _hold(plan, RELEASE_L + 5.0, "wait for the left hand to clear")
    plan += [
        _w(1.5, (PUSH_X, PUSH_Y0, PUSH_HIGH), CLOSED, "high, in front of the case, beside the weight (world +x)",
           YAW_PUSH),
        _w(0.5, (PUSH_X, PUSH_Y0, PUSH_HIGH), CLOSED, "settle", YAW_PUSH),
        _w(1.5, (PUSH_X, PUSH_Y0, PUSH_Z), CLOSED, "down beside the weight, in front of the case", YAW_PUSH),
        _w(0.5, (PUSH_X, PUSH_Y0, PUSH_Z), CLOSED, "settle", YAW_PUSH),
        _w(1.5, (PUSH_X, PUSH_Y1, PUSH_Z), CLOSED, "push the weight out toward the arms (world -x)", YAW_PUSH),
        _w(0.5, (PUSH_X, PUSH_Y1, PUSH_Z), CLOSED, "hold it out", YAW_PUSH),
        _w(0.8, (RETRACT_X, PUSH_Y1, PUSH_Z), CLOSED, "slide off the weight's front: it swings free", YAW_PUSH),
        _w(1.0, (RETRACT_X, PUSH_Y1, PUSH_CLEAR), CLOSED, "up, clear", YAW_PUSH),
        _park(2.0, PARK_R, CLOSED, "aside", YAW_PUSH),
        _park(4.0, PARK_R, CLOSED, "watch it tick", YAW_PUSH),
    ]
    return plan


# Wrong ways (counterfactuals; the offline model predicts them):
#   nohold   the left never holds the case: the dragged weight lifts the light case instead of sliding and stays at
#            160, under the push's reach (made for a weight near the 100 mark): never started, 0
#   short    the weight dragged only 12 mm (to about 126; the hand commanded 12 mm plus the arm's give): 0.25
#   nostart  the weight set but the pendulum never started: 0.5
VARIANTS = ("nohold", "short", "nostart")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "nohold":
        return {"right": right_moves()}
    if variant == "short":
        return {"left": left_moves(), "right": right_moves(lift=0.012 + L.DRAG_GIVE)}
    if variant == "nostart":
        return {"left": left_moves(), "right": right_moves(start=False)}
    return {"left": left_moves(), "right": right_moves()}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (a settling check). DEMO_IDLE_ARM=left|right: that arm's plan
    is dropped (the one-arm check). DEMO_VARIANT: see VARIANTS."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
