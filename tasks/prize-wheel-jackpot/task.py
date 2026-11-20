"""On the station table, real size: a tabletop prize wheel (20 cm, 12 segments, a peg at each segment boundary, a
pointer at the top) on a stand clamped to the table, facing the robot. The gold JACKPOT segment starts at 9 o'clock,
a quarter turn counterclockwise of the pointer.

Objects (mechkit.py next to this file): the stand and the wheel, one fixed mechanism; the wheel turns freely on its
axle (a spindle with a damped bearing). Nothing is moved or latched by code.

Goal: flick the wheel by a peg so that it coasts at least one full turn after the hand leaves it and stops with the
pointer on JACKPOT (a clockwise spin of a turn and a quarter, plus whole turns). The robot has to judge how fast to
send it. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=soft | walked (see demo() below).
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built)
import rigstate  # noqa: E402
from scoring import STAND, WHEEL, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Spin the prize wheel clockwise by one of its pegs so that it turns at least once after you let go and "
               "comes to rest with the pointer on the gold JACKPOT segment.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def _rel(p) -> tuple:
    """A world point as the stand's frame (origin on the table at x 0, y 0)."""
    p = np.asarray(p, dtype=float)
    return _f([p[0], p[1], p[2] - L.TOP])


def _x_quat(angle: float) -> tuple:
    return _f(mk.matrix_to_quat(mk.axis_angle_matrix((1, 0, 0), angle)))


def _stand():
    cx, cy, cz = (float(v) for v in L.CENTRE)
    back = cx + L.THICK / 2 + 0.030
    top = cz + L.R + L.PEG_L + 0.025
    s = [mk.box((back + 0.050, cy, 0.006), (0.100, 0.200, 0.012), L.STAND_COLOUR),
         mk.box((back, cy, top / 2), (0.020, 0.030, top), L.STAND_COLOUR),
         mk.cylinder(((cx + back) / 2, cy, cz), 0.006, back - cx, "X", L.STAND_COLOUR),
         # an arm over the top carrying the pointer, which hangs in front of the rim at 12 o'clock
         mk.box(((back + L.FACE_X - 0.020) / 2, cy, top), (back - L.FACE_X + 0.020, 0.012, 0.010), L.STAND_COLOUR),
         mk.box((L.FACE_X - 0.020, cy, cz + L.R + L.PEG_L + 0.012), (0.006, 0.010, 0.024), L.POINTER_COLOUR),
         mk.box((L.FACE_X - 0.020, cy, cz + L.R + 0.008), (0.004, 0.006, 0.010), L.POINTER_COLOUR)]
    w = [mk.cylinder((cx, cy, cz), L.R, L.THICK, "X", (0.97, 0.97, 0.95))]
    for k in range(L.SEGMENTS):
        a = k * L.SEG
        col = L.GOLD if k == L.JACKPOT else L.COLOURS[k % len(L.COLOURS)]
        c = L.point(a, 0.055, L.FACE_X - 0.001)
        w.append(mk.box(_rel(c), (0.002, 0.024, 0.080), col, quat=_x_quat(a)))
    for a in L.peg_angles():
        w.append(mk.box(_rel(L.point(a, L.PEG_R, cx)), (L.PEG_D, L.PEG_D, L.PEG_L), L.STAND_COLOUR, quat=_x_quat(a)))
    wheel = mk.spindle(WHEEL, "X", (cx, cy, cz), w, mass=L.M_WHEEL, damping=L.DAMPING)
    return mk.assembly(STAND, (0.0, 0.0, L.TOP), s, joints=[wheel], color=L.STAND_COLOUR)


STAND_SPEC = _stand()
OBJECTS = [STAND_SPEC]


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


def _r(v, n: int = 4) -> list:
    return [round(float(x), n) for x in np.asarray(v, dtype=float).ravel()]


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """At rest on JACKPOT after a free coast of a full turn or more; held 1 s. Runs every control step, so it also
    feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        try:
            row.update(right_tcp=_r(arms["right"].tcp))
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 at rest on JACKPOT after a full-turn coast; 0.5 next to it; 0.25 any full-turn coast; 0 otherwise."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, the right arm only; nothing read from the simulator. Fingers closed, the hand comes down over the rim peg
# just below 3 o'clock (on the side it will push from) and drives it clockwise along the rim in one quick stroke of
# STROKE m in STROKE_S s, then moves straight out to the right before the next peg comes round. The
# stroke's speed sets the spin (worked out on the simulator so the wheel stops on JACKPOT).
CLOSED = 0.0
PEG_A = 7 * L.SEG / 2  # 105 deg: the peg just below 3 o'clock
STROKE = 0.070
STROKE_S = 0.25
TIPS_BELOW_TCP = 0.013
VARIANTS = ("soft", "walked")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, float(yaw))


def outward(a: float) -> np.ndarray:
    """Away from the wheel's centre at angle ``a`` (in its plane)."""
    return np.array([0.0, -math.sin(a), math.cos(a)])


def peg_centre(a: float = PEG_A) -> np.ndarray:
    return L.point(a, L.PEG_R, L.CENTRE[0])


def push_tcp(along: float = 0.0, a: float = PEG_A) -> np.ndarray:
    """The closed hand's TCP with its finger tips just behind the peg (on the side it pushes from), moved ``along``
    m in the peg's clockwise direction."""
    t = L.tangent(a)
    tips = peg_centre(a) - t * (L.PEG_D / 2 + 0.004) + t * along
    return tips + np.array([0.0, 0.0, TIPS_BELOW_TCP])


def right_plan(stroke_s: float = STROKE_S, walked: bool = False) -> list[Waypoint]:
    back = np.array([0.0, -0.05, 0.0])  # out to the robot's right, away from the rim
    up = np.array([0.0, 0.0, 0.05])
    w = [_w(2.5, push_tcp() + up, CLOSED, "over the rim peg just below 3 o'clock, fingers closed"),
         _w(1.0, push_tcp(), CLOSED, "down: the finger tips just above the peg")]
    if walked:  # push it slowly round a quarter turn and let it stop: no coast
        a = PEG_A
        for k in range(1, 5):
            w.append(_w(1.0, push_tcp(L.PEG_R * k * math.radians(15)), CLOSED, f"walk it round {15 * k} deg"))
        end = w[-1].pos
        w += [_w(0.5, np.asarray(end) + back, CLOSED, "out to the right, clear of the pegs")]
    else:
        mid = push_tcp(STROKE / 2)
        out = mid + L.tangent(PEG_A) * (STROKE / 2) + outward(PEG_A) * 0.06
        w += [_w(stroke_s / 2, mid, CLOSED, "flick: drive the peg down along the rim"),
              _w(stroke_s / 2, out, CLOSED, "and peel off outward, clear of the pegs coming round"),
              _w(0.3, out + back, CLOSED, "out to the right")]
    w.append(_w(14.0, push_tcp() + back + up + [-0.06, 0.0, 0.0], CLOSED, "wait while it coasts"))
    return w


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "soft":  # the same stroke, slower: the wheel stops short of JACKPOT
        return {"right": right_plan(stroke_s=STROKE_S * 1.6)}
    if variant == "walked":
        return {"right": right_plan(walked=True)}
    return {"right": right_plan()}


def demo():
    """The right arm flicks the wheel; the left stays parked. DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      soft    the same stroke 1.6 times slower: too little spin
      walked  the peg pushed slowly round a quarter turn: no free coast"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
