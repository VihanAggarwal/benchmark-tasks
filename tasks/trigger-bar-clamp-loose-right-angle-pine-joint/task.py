"""On the station table, real size: two loose pieces of pine cut from 1x2 furring (19 x 38 mm), lying flat as a person leaves
them after a rough dry fit of a small frame's corner (rail B, 300 mm, along y; stub A, 100 mm, along x with its end 4 mm off
B's face, 8 mm short of flush with B's end and 2.5 degrees out of square), and an IRWIN QUICK-GRIP Mini one-handed bar clamp,
6 in (a steel bar with a fixed jaw, a moving jaw with a pistol grip, a ratchet and a release tab), lying on its side on the
bench, open, hung loosely over the corner (3 mm clear behind the rail, 9 mm beyond the stub), its outer side 2.5 mm off a bench stop (a pine strip fixed to the bench).

Goal: the stub's end against the rail's face, its outer face flush with the rail's end, square (within 1.4 degrees), the clamp
over the corner with its pads on the rail's back face and the stub's free end and at least 40 N on them, both hands off.

The clamp is a mechkit assembly (mechkit.py, next to this file): a fixed jaw and bar (the base), the moving jaw on a slider
(a spring to a target), the trigger and the release tab on hinges of their own. The ratchet is the one declared rule: each
full squeeze of the trigger moves the slider's target one 3 mm step toward the fixed jaw (the pawl); the pull the target
makes past the jaw is the clamp force (20 N per mm); the release tab sets the target back to 0. The boards, the pads and the
gripper's pads are plain contacts. The arms are read through rigstate.py (the let-go check).

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
import offline_clamp  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Clamp the two pine boards together at the corner, flush and square, with the trigger bar clamp, and let "
               "go of everything.")

# -- the parts --------------------------------------------------------------------------------------------------------
B_PART = mk.assembly(S.B, origin=(L.B_CX, L.B_CY, L.TOP), shapes=P.b_shapes(), mass=L.B_MASS, color=L.PINE,
                     friction=L.PINE_FRICTION)
A_PART = mk.assembly(S.A, origin=(L.A_CX + L.A_START_DX, L.A_CY + L.A_START_DY, L.TOP), shapes=P.a_shapes(),
                     mass=L.A_MASS, yaw=L.A_START_YAW, color=L.PINE, friction=L.PINE_FRICTION)
CLAMP_PART = mk.assembly(S.CLAMP, origin=(L.CLAMP_XF, L.CLAMP_YC0, L.TOP), shapes=P.base_shapes(), mass=0.15,
                         joints=P.clamp_joints(), color=L.CLAMP_YELLOW, friction=L.CLAMP_FRICTION, tight_fit=True)
STOP_PART = mk.assembly("bench_stop", origin=((L.STOP_X0 + L.STOP_X1) / 2, (L.STOP_Y0 + L.STOP_Y1) / 2, L.TOP),
                        shapes=P.stop_shapes(), color=L.PINE, friction=L.PINE_FRICTION)  # no mass: fixed to the bench
OBJECTS = [B_PART, A_PART, CLAMP_PART, STOP_PART]

_OFFLINE = offline_clamp.ClampModel(A_PART, B_PART, CLAMP_PART)

# -- the ratchet (the one rule) -----------------------------------------------------------------------------------------
RATCHET = {"pull": 0.0, "armed": True, "steps": 0, "freed": 0}


def _reset_ratchet() -> None:
    RATCHET.update(pull=0.0, armed=True, steps=0, freed=0)


@mk.on_step
def _ratchet(states) -> None:
    """The pawl. A squeeze past 80% of the trigger's travel pulls the moving jaw one step (3 mm) toward the fixed jaw, once
    until the trigger has been back under 30%; the release tab turned past 60% frees the pawl while the trigger is not squeezed (the pull goes to 0). The
    target is written every step (mechkit's targets go back to 0 at reset)."""
    c = states.get(S.CLAMP)
    try:
        trig, rel = float(c.joints["trigger"]), float(c.joints["release"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    f = min(max(trig / L.TRIG_MAX, 0.0), 1.0)
    if rel >= L.REL_ON * L.REL_UPPER and f < L.TRIG_ON:  # the tab frees the pawl when the trigger is not squeezed
        RATCHET["pull"] = 0.0
        RATCHET["freed"] += 1
    elif f >= L.TRIG_ON and RATCHET["armed"]:
        RATCHET["pull"] = min(RATCHET["pull"] + L.STEP, L.JAW_UPPER)
        RATCHET["steps"] += 1
        RATCHET["armed"] = False
    elif f <= L.TRIG_OFF:
        RATCHET["armed"] = True
    mk.set_targets(CLAMP_PART, "jaw", RATCHET["pull"])


def no_simulator(parts) -> bool:
    return mk.env() is None and getattr(parts.get(S.CLAMP), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def _fresh() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _reset_ratchet()


@rigstate.on_reset
def _new_episode() -> None:
    _fresh()


@mk.on_reset
def _kit_reset() -> None:
    _fresh()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the parts' poses and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Clamped (pads on both boards, at least 40 N), the joint within the tight limits, both hands clear and everything
    still for 1 s, no latch. This runs first every control step, so it also feeds the tracker (and, without a simulator,
    the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _fresh()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
        pull = _OFFLINE.target
    else:
        pull = RATCHET["pull"]
    result = _TRACKER.update(parts, arms, pull)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, "ratchet": dict(RATCHET), **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(S.CLAMP), "source", None)
    try:  # the clamp's own pose (the GPU run of 2026-11-20 could not show that it had turned): origin in mm, yaw in deg
        cp = S.read_pose(parts[S.CLAMP])
        row["clamp_pose"] = [round(float(cp.pos[0]) * 1000, 1), round(float(cp.pos[1]) * 1000, 1),
                             round(math.degrees(S.yaw(cp.rot)), 2)]
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if offline:
        row["offline"] = {"holds": dict(_OFFLINE.holds), "target_mm": round(_OFFLINE.target * 1000, 2)}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the clamp is on both boards with at least 40 N (and nothing fell off the table); 0.25 clamped, the joint
    outside the loose limits (gap 3, flush 3, square 5 mm); 0.5 inside those, outside the tight ones (1, 1.5, 2.5 mm); 0.75
    tight, but a hand still within 3 cm of anything or something moved in the last second; 1.0 tight, hands clear, still.
    The latest scored step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. Straight-down grips; finger
# targets: the joint value is minus half the commanded gap. Left: closes along y (yaw 0) on stub A. Right: closes along x
# (yaw +pi/2) on the clamp's pistol grip: the pads on the handle's far face and on the trigger's far face.
# The clamp is NOT carried: the GPU run of 2026-11-20 dragged it 70 mm by the grip and it turned 10.5 degrees about the
# pinch (a pinch holds no yaw torque; a push at the grip, 85 mm from the clamp's centre of friction, swings it round) and
# lagged the hand by 43 mm, so the jaws never reached the boards. It starts over the corner and the hand only squeezes.
PAD = 0.013  # the pad tips are this far past the TCP along the approach axis
YAW_A = 0.0
YAW_C = math.pi / 2
TIP_UP = 0.008  # pad tips 8 mm over the bench: under A's top (19 mm), over the handle's middle
U = TIP_UP + PAD  # TCP height over the bench
HOVER = 0.080


def fingers(gap: float) -> float:
    return -gap / 2


A_AROUND = fingers(0.052)  # open round A (38 mm wide, its centre 8 mm off the gripper's axis): the outer pad stays 9 mm
#                            clear of the clamp's bar (a wider opening would put the pad on the bar)
A_GRIP = 0.0  # commanded shut: the pads close on A and square it
A_LETGO = fingers(0.046)  # opened round A: the outer pad's face stays 4 mm clear of the clamp's bar
C_AROUND = fingers(0.072)  # open round the grip (64 mm pad to pad with the trigger at rest)
C_OPEN = fingers(0.070)  # between squeezes: 3 mm clear each side, the trigger springs back
C_SQUEEZE = 0.0  # commanded shut: the pads press to the trigger's stop (46 mm)
C_LETGO = fingers(0.080)
SQUEEZES = 6  # 4 to take up the 12 mm of slack, 2 loading it: 120 N at the end
PARK = {"left": (0.20, 0.27, 0.16), "right": (0.20, -0.27, 0.16)}
T_SQUEEZE = 1.5 + 0.5 + 1.5  # close, hold, open
T_WAIT = 8.0  # the right hand takes hold of the grip only after the left has closed, squared and pressed the stub (7 s)


def _w(d, p, finger, label, yaw):
    return Waypoint(d, (float(p[0]), float(p[1]), float(L.TOP + p[2])), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _hold(plan, until, label):
    gap = until - _duration(plan)
    if gap > 1e-9:
        last = plan[-1]
        plan.append(Waypoint(gap, last.pos, last.finger, label, last.yaw))


def grip_xy(x_jaw: float = 0.0, y_clamp: float = L.CLAMP_YC0):
    """The right gripper's centre on the clamp's pistol grip with the jaw pulled ``x_jaw`` and the bar at ``y_clamp``."""
    return L.CLAMP_XF + L.GRIP_LX - x_jaw, y_clamp + L.GRIP_LY


def left_moves(until_release: float) -> list:
    """Over stub A, down either side of it, close (the pads centre and square it), press it 4 mm in against B's face and
    hold it while the clamp goes on; open, up, away. The grip is on A's end nearer B (the pads clear the moving pad by 24 mm)."""
    x, y = L.A_HOLD_X + L.A_START_DX, L.A_CY  # the gripper's axis on the flush line; A is 8 mm off it
    y_a = L.A_CY + L.A_START_DY  # A's centre: the pads go down round it (the outer pad stays clear of the clamp's bar)
    plan = [
        _w(2.0, (x, y_a, HOVER), A_AROUND, "over stub A, fingers 52 mm apart", YAW_A),
        _w(1.5, (x, y_a, U), A_AROUND, "down either side of A", YAW_A),
        _w(0.5, (x, y_a, U), A_AROUND, "settle", YAW_A),
        _w(1.5, (x, y, U), A_GRIP, "close: the pads centre A on the flush line and square it", YAW_A),
        _w(0.5, (x, y, U), A_GRIP, "settle the pinch", YAW_A),
        _w(1.0, (L.A_HOLD_X - 0.0005, y, U), A_GRIP, "press A's end against B's face", YAW_A),
    ]
    _hold(plan, until_release, "hold A square and flush while the clamp goes on")
    plan += [
        _w(1.5, plan[-1].pos[:2] + (plan[-1].pos[2] - L.TOP,), A_LETGO, "open", YAW_A),
        _w(1.0, (L.A_HOLD_X - 0.0005, y, HOVER), A_LETGO, "straight up", YAW_A),
        _w(2.0, PARK["left"], A_LETGO, "away", 0.0),
    ]
    return plan


def right_moves(squeezes: int = SQUEEZES) -> list:
    """Over the clamp's pistol grip (the clamp lies over the corner already), down round the handle and the trigger,
    wait for the left hand, then ``squeezes`` full squeezes (the hand follows the handle as the jaw advances), open, up,
    away."""
    gx, gy1 = grip_xy(0.0, L.CLAMP_YC)
    plan = [
        _w(3.0, (gx, gy1, HOVER), C_AROUND, "over the clamp's pistol grip, fingers 72 mm apart", YAW_C),
        _w(1.5, (gx, gy1, U), C_AROUND, "down round the handle and the trigger", YAW_C),
        _w(0.5, (gx, gy1, U), C_AROUND, "settle", YAW_C),
    ]
    _hold(plan, T_WAIT, "wait until the left hand has squared and flushed the stub")
    for k in range(squeezes):
        x_now, _ = grip_xy(L.STEP * min(k, 4), L.CLAMP_YC)
        plan += [
            _w(1.5, (x_now, gy1, U), C_SQUEEZE, f"squeeze {k + 1}: close on the grip, the jaw takes one step", YAW_C),
            _w(0.5, (x_now, gy1, U), C_SQUEEZE, "hold", YAW_C),
        ]
        x_next, _ = grip_xy(L.STEP * min(k + 1, 4), L.CLAMP_YC)
        if k < squeezes - 1:
            plan.append(_w(1.5, (x_next, gy1, U), C_OPEN, "open: the trigger springs back", YAW_C))
        else:
            plan.append(_w(1.5, (x_next, gy1, U), C_LETGO, "open: the pawl holds the jaw", YAW_C))
    xe, _ = grip_xy(L.STEP * 4, L.CLAMP_YC)
    plan += [
        _w(1.0, (xe, gy1, HOVER), C_LETGO, "straight up", YAW_C),
        _w(2.0, PARK["right"], C_LETGO, "away", 0.0),
    ]
    return plan


def park_moves(side: str, duration: float) -> list:
    return [_w(2.5, PARK[side], A_LETGO, "away", 0.0), _w(max(duration - 2.5, 0.1), PARK[side], A_LETGO, "hold", 0.0)]


# Wrong ways (counterfactuals; the offline model predicts them):
#   four   only four squeezes: the pad just touches, the pull equals the jaw's position, no force: 0
#   five   five squeezes: 60 N (the demo loads 120 N): still clamped, 1.0
VARIANT_SQUEEZES = {"four": 4, "five": 5}
VARIANTS = tuple(VARIANT_SQUEEZES)


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    n = VARIANT_SQUEEZES.get(variant, SQUEEZES)
    right = right_moves(n)
    t_right = _duration(right)
    # the left hand lets go when the right hand has opened after its last squeeze (the right's plan up to "straight up")
    t_release = _duration(right[:-2])
    return {"left": left_moves(t_release), "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check). DEMO_VARIANT: see VARIANTS."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
