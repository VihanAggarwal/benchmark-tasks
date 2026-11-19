"""On the station table, real size: a child-size polystyrene head form (52 cm round) faces the arms; in front of it a
call headset (Jabra Evolve2 30 SE stereo class, 102 g) hangs on its desk stand, its microphone boom stowed straight
up. The child is about to join an online class.

Goal: the microphone boom lowered to its stop in front of the mouth, the headset on the head with each ear cup on its
own ear (the boom's cup on the child's left), both hands away.

Physics only (mechkit.py, next to this file): the headset is one loose articulation. Its base is the headband; each
cup rides on a sprung slide along y (the band's clamp, 2 N per side on the ears), so the cups must be pulled apart to
pass over the ears; the boom turns on a hinge on the left cup, held up by gravity against its upper stop and down by
gravity against its lower stop. The head and the stand are fixed. No rule and no flag: the score reads poses only.

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
import offline_headset  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (rolled grips for the demo: wraps yam_arena.demo.grip_rotation)
from scoring import HEADSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Put the headset on the child-size mannequin head: lower its microphone boom, then put it on with each ear cup on its own ear, the microphone's cup on the head's left, and let go."

HEAD, STAND = "head_form", "headset_stand"
HEAD_PART = mk.assembly(HEAD, origin=L.HEAD_ORIGIN, shapes=P.head_shapes(), color=L.FOAM)
STAND_PART = mk.assembly(STAND, origin=L.STAND_ORIGIN, shapes=P.stand_shapes(), color=L.SILVER)
HEADSET_PART = mk.assembly(HEADSET, origin=L.HEADSET_ORIGIN_START, shapes=P.band_shapes(), mass=L.BAND_MASS,
                           joints=P.joints(), color=L.BLACK, friction=L.HEADSET_FRICTION)
OBJECTS = [HEAD_PART, STAND_PART, HEADSET_PART]

_OFFLINE = offline_headset.HeadsetModel(HEADSET_PART, HEAD_PART, STAND_PART)


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(HEADSET), "source", "sim") != "sim"


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
    """Seated, on both ears, microphone down, resting and hands off on every step of the last 1 s, no latch. Runs
    first every control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
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
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row["joints"] = parts[HEADSET].joints
        row["headset_origin"] = parts[HEADSET].origin
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
    if offline:
        row["offline"] = {"mode": _OFFLINE.mode, "holds": dict(_OFFLINE.holds), "slips": _OFFLINE.slips}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the headset fell or is still on its stand; 0.25 off the stand, not seated on both ears; 0.5 seated on both
    ears, microphone up; 0.75 seated, on the ears, microphone down, but not resting with both hands away for the
    whole last second; 1.0 all of it. The latest scored step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both cup grips close along
# world x (yaw -pi/2 left, +pi/2 right) and are rolled about x (tiltgrip.py) so each wrist leans outward, its housing
# clear of the headband over the cup: the left (on the +y cup) -0.75 rad (pad tips toward -y), the right +0.75 rad.
# Points below are pad-tip points (table frame); _w turns them into TCP targets 13 mm back along the approach axis.
YAW_L, YAW_R = -math.pi / 2, math.pi / 2
ROLL = 0.75
ROT = {"left": tiltgrip.tilted(YAW_L, -ROLL), "right": tiltgrip.tilted(YAW_R, ROLL), "push": YAW_R}
APP = {"left": tiltgrip.approach(YAW_L, -ROLL), "right": tiltgrip.approach(YAW_R, ROLL),
       "push": np.array([0.0, 0.0, -1.0])}
PAD_TIP = 0.013
BACK = 0.050  # hover: this far back along the approach axis from the grip point


def fingers(gap: float) -> float:
    return -gap / 2


F_OPEN, F_GRIP, F_PUSH = fingers(L.OPEN_GAP), fingers(L.GRIP_GAP), fingers(L.PUSH_GAP)
SHELL_YC = offline_headset.SHELL_YC
CUP_Z = L.STAND_TOP + L.CUP_ZC  # cup centre on the stand: 122 mm
GRIP_Z = CUP_Z + L.GRIP_DZ
LIFT = 0.030
SEAT_CUP_Z = L.SEATED_Z + L.CUP_ZC  # 115 mm
HOVER_DROP = 0.003  # the band 3 mm over the crown when the cups close in
# The push pads ride 3 mm outside the boom arm's centre line so the 27 mm wide fingers stay clear of the band's side
# (y -75 to -95 mm) while they still cover the 6 mm arm (y -105 to -111 mm).
PUSH = {"y": L.BOOM_Y - 0.003, "z": 0.185, "x0": L.SX + 0.035, "x1": L.SX + 0.005}
PARK = {"left": (0.200, 0.250, 0.170), "right": (0.200, -0.250, 0.170)}


def _w(d, tip, finger, label, arm, back=0.0):
    a = APP[arm]
    t = np.array([tip[0], tip[1], L.TOP + tip[2]]) - (PAD_TIP + back) * a
    return Waypoint(d, (float(t[0]), float(t[1]), float(t[2])), finger, label, ROT[arm])


def _park(d, xyz, finger, label, arm):
    return Waypoint(d, (xyz[0], xyz[1], L.TOP + xyz[2]), finger, label, ROT[arm])


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _hold(plan, until, label):
    last = plan[-1]
    gap = until - _duration(plan)
    if gap > 1e-9:
        plan.append(Waypoint(gap, last.pos, last.finger, label, last.yaw))


T_DESCEND = 9.0


def cup_moves(arm: str, spread: float = L.SPREAD_CMD, seat_in: float = L.Q_SEAT) -> list:
    """Over its cup on the stand, down to it, grip it across its 60 mm (front to back), pull it out, lift, carry it
    over the head, down to the ears, let the cup in onto the ear, the band onto the crown, open, back out, aside."""
    s = 1.0 if arm == "left" else -1.0
    y0, x0 = L.SY + s * SHELL_YC, L.SX
    plan = [_w(3.0, (x0, y0, GRIP_Z), F_OPEN, "over its cup, fingers 86 mm apart", arm, back=BACK)]
    _hold(plan, T_DESCEND, "wait")
    ys, yseat = L.SY + s * (SHELL_YC + spread), L.HY + s * (SHELL_YC + seat_in)
    zc = SEAT_CUP_Z + L.GRIP_DZ
    plan += [
        _w(1.5, (x0, y0, GRIP_Z), F_OPEN, "down either side of the cup", arm),
        _w(0.5, (x0, y0, GRIP_Z), F_OPEN, "settle", arm),
        _w(1.5, (x0, y0, GRIP_Z), F_GRIP, "grip the cup front to back", arm),
        _w(0.5, (x0, y0, GRIP_Z), F_GRIP, "settle the grip", arm),
        _w(1.5, (x0, ys, GRIP_Z), F_GRIP, "pull the cup out 30 mm against the band", arm),
        _w(2.0, (x0, ys, GRIP_Z + LIFT), F_GRIP, "lift the headset 30 mm off the stand", arm),
        _w(4.0, (L.HX, L.HY + s * (SHELL_YC + spread), GRIP_Z + LIFT), F_GRIP, "carry it over the head, cups "
           "outside the ears, band over the crown", arm),
        _w(2.5, (L.HX, L.HY + s * (SHELL_YC + spread), zc + HOVER_DROP), F_GRIP, "down to ear height", arm),
        _w(1.5, (L.HX, yseat, zc + HOVER_DROP), F_GRIP, "let the cup in onto the ear", arm),
        _w(0.5, (L.HX, yseat, zc + HOVER_DROP), F_GRIP, "settle", arm),
        _w(1.0, (L.HX, yseat, zc), F_GRIP, "the band down onto the crown", arm),
        _w(0.5, (L.HX, yseat, zc), F_GRIP, "settle", arm),
        _w(1.5, (L.HX, yseat, zc), F_OPEN, "let go", arm),
        _w(0.5, (L.HX, yseat, zc), F_OPEN, "settle", arm),
        _w(1.0, (L.HX, yseat, zc), F_OPEN, "back out along the wrist", arm, back=BACK),
        _park(2.0, PARK[arm], F_OPEN, "aside", arm),
    ]
    return plan


def boom_moves() -> list:
    """Right arm first: fingers closed, behind the stowed boom, push it forward past upright; it falls to its lower
    stop. Then back and over to its cup (the rest of cup_moves)."""
    y, z, x0, x1 = PUSH["y"], PUSH["z"], PUSH["x0"], PUSH["x1"]
    return [
        _w(3.0, (x0, y, z + 0.015), F_PUSH, "behind the stowed boom, fingers closed", "push"),
        _w(1.0, (x0, y, z), F_PUSH, "down beside its arm", "push"),
        _w(2.0, (x1, y, z), F_PUSH, "push the boom forward past upright: it falls to its stop", "push"),
        _w(0.5, (x1, y, z), F_PUSH, "settle", "push"),
        _w(1.0, (x0 + 0.010, y, z + 0.015), F_PUSH, "back and up", "push"),
    ]


def right_moves(push: bool = True) -> list:
    plan = boom_moves() if push else []
    rest = cup_moves("right")
    first = rest[0]
    plan.append(Waypoint(1.5, first.pos, F_OPEN, first.label, first.yaw))
    _hold(plan, T_DESCEND, "wait")
    return plan + rest[2:]


# Wrong ways (counterfactuals; the offline model predicts them):
#   nopush   the boom is never lowered: seated on both ears, microphone up: 0.5
#   nospread the cups are not pulled apart: the cushions meet the ears on the way down and the grips slip: 0
VARIANTS = ("nopush", "nospread")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "nopush":
        return {"left": cup_moves("left"), "right": right_moves(push=False)}
    if variant == "nospread":
        r = right_moves()
        return {"left": cup_moves("left", spread=0.0, seat_in=0.0),
                "right": r[:6] + cup_moves("right", spread=0.0, seat_in=0.0)[2:]}
    return {"left": cup_moves("left"), "right": right_moves()}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan. DEMO_IDLE_ARM=left|right: that arm's plan is dropped.
    DEMO_VARIANT: see VARIANTS."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
