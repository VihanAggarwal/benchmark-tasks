"""On the station table, real size: a loose molded-pulp four-cup drink carrier (196 mm square, 43 mm to the pocket
mouths, 0.10 kg) stands 22 cm out between the arms, and four lidded 8 oz paper hot cups (tapered, 56 mm base, 80 mm rim,
an 82 mm press-on lid, 95 mm tall, 0.25 kg, filled) stand beside it, two a side, 34 cm out from the midline. Each
pocket is a 12-sided tapered funnel 76 mm across at its mouth and 62 mm at its floor, 36 mm deep, on a 104 mm pitch.

Goal: every cup upright and bottomed on the floor of its own pocket, both hands off, the carrier where it was.

The carrier is one mechkit compound (mechkit.py, next to this file): a base plate and four funnels of 12 slabs each,
loose. Each cup is a loose compound too: 11 stacked discs for its cone, the lid's skirt and its dome. No software
rules: the pockets are geometry. The arms are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Trace: TASK_TRACE=<file> (one JSON line per
control step).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the carrier is built as a compound)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "Stand all four cups upright in the four pockets of the carrier, then take both hands off and leave the "
    "carrier where it is."
)


# -- the parts --------------------------------------------------------------------------------------------------------
def carrier_shapes():
    """The base plate (7 mm, the pockets' floors) and four funnels of 12 slabs. Base frame: on the table at the
    carrier's centre, z up."""
    out = [mk.box((0.0, 0.0, L.FLOOR_TOP / 2), (L.PLATE, L.PLATE, L.FLOOR_TOP), L.PULP_COLOR)]
    for x, y in L.POCKET_XY.values():
        for centre, size, ypr in L.funnel_slabs(x - L.CAR_X, y - L.CAR_Y, L.POCKET_R_RIM, L.POCKET_R_FLOOR,
                                                L.POCKET_D, L.FLOOR_TOP):
            out.append(mk.box(centre, size, L.PULP_COLOR, ypr=ypr))
    return out


CARRIER_PART = mk.assembly(S.CARRIER, origin=(L.CAR_X, L.CAR_Y, L.TOP), mass=L.CAR_MASS, shapes=carrier_shapes(),
                           color=L.PULP_COLOR, friction=L.CAR_FRICTION, tight_fit=True)


def cup_shapes():
    """The tapered 8 oz cup (11 stacked discs), its lid's skirt and dome. Cup frame: origin at its underside, z up."""
    return [mk.cylinder((0.0, 0.0, zc), d / 2, h, "Z", L.LID_COLOR if kind == "lid" else L.CUP_COLOR)
            for kind, zc, d, h in L.cup_parts()]


# Each cup's frame is turned to its arm's grip yaw (a round cup looks the same either way; the offline checker draws
# a compound as its yawed box, so this lines that box up with the fingers).
CUP_YAW = {"cup_a": math.pi / 4, "cup_b": math.pi / 4, "cup_c": -math.pi / 4, "cup_d": -math.pi / 4}


def _cup(name):
    x, y = L.CUP_START[name]
    return mk.assembly(name, origin=(x, y, L.TOP), mass=L.CUP_MASS, shapes=cup_shapes(), yaw=CUP_YAW[name],
                       color=L.CUP_COLOR, tight_fit=True)


OBJECTS = [CARRIER_PART] + [_cup(n) for n in L.CUP_NAMES]


# -- the offline stand-in for the physics -----------------------------------------------------------------------------
# The checker's proxy carries a gripped cup rigidly and drops a released one onto the carrier's bounding box (a block
# 55 mm tall), or lets it fall to the table. Neither is a cup going down a funnel to the pocket's floor. Offline only
# (never in the simulator) this model puts any cup whose axis is within 8 mm of a pocket's and whose underside is below
# the mouth on that pocket's floor, as the funnel would.
SNAP_XY = L.SEAT_XY_TOL


def no_simulator(states) -> bool:
    return mk.env() is None and getattr(states.get(S.CARRIER), "source", "sim") != "sim"


def offline_seat(states: dict) -> dict:
    out = dict(states)
    for c in L.CUP_NAMES:
        s = out.get(c)
        if s is None:
            continue
        base = np.array(s.origin if getattr(s, "origin", None) is not None else
                        np.asarray(s.pos, dtype=float) - np.array([0.0, 0.0, L.CUP_H / 2]), dtype=float)
        bottom = base[2] - L.TOP
        if not (-0.001 <= bottom <= L.RIM_H + 0.003):
            continue
        for x, y in L.POCKET_XY.values():
            if math.hypot(base[0] - x, base[1] - y) <= SNAP_XY:
                shift = np.array([0.0, 0.0, L.TOP + L.FLOOR_TOP - base[2]])
                fields = {"pos": np.asarray(s.pos, dtype=float) + shift, "low": np.array(s.low) + shift,
                          "high": np.array(s.high) + shift}
                if getattr(s, "origin", None) is not None:
                    fields["origin"] = base + shift
                if dataclasses.is_dataclass(s):
                    out[c] = dataclasses.replace(s, **fields)
                else:  # a test double
                    out[c] = types.SimpleNamespace(**{**vars(s), **fields})
                break
    return out


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: every part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """All four cups seated, both grippers 30 mm clear and every part still for 1 s, no latch; held 1 s. This runs
    first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    if no_simulator(parts):
        parts = offline_seat(parts)
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
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
    """0 if a cup tilted more than 15 degrees or fell faster than 0.886 m/s at any step, the carrier moved over 30 mm or
    tilted over 5 degrees, something fell off the table, a cup ends lying, or unless every part is let go and still;
    otherwise the number of pockets with a seated cup over four. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
GRIP_ON = 0.047  # the pad tips this far over the cup's underside: the 38 mm pads span 47 to 85 mm, under the lid
GRIP_D = max(d for kind, zc, d, h in L.cup_parts() if kind == "body" and zc - h / 2 < GRIP_ON + 0.038)  # 78.9 mm
GRIP = -(GRIP_D - 0.008) / 2  # 70.9 mm commanded across the cup's 78.9 mm near its rim
GRIP_UP = GRIP_ON + PAD_TIP  # the TCP this far above the cup's underside
PICK_Z = GRIP_UP  # the TCP's height over the table when the cup stands on it
CARRY_BOTTOM = L.RIM_H + 0.023  # commanded underside height while carrying: 23 mm over the mouths (the arm sags)
CARRY_Z = CARRY_BOTTOM + GRIP_UP
ABOVE_MOUTH = L.RIM_H + 0.016 + GRIP_UP  # the underside 16 mm over the mouth (about 9 mm once the arm has sagged)
SEAT_Z = L.FLOOR_TOP + GRIP_UP  # the TCP when the cup's underside is on the pocket's floor
PRESS = 0.004  # commanded this far lower: the arm sags under the cup's 2.5 N and unloads once it rests
HOVER = 0.170
YAW = {"left": math.pi / 4, "right": -math.pi / 4}  # the fingers close along a diagonal of the pocket grid
OVERLAP = 2.0  # one arm starts its approach this long before the other has cleared the carrier


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def pick_moves(arm, cup, first: bool):
    x, y = L.CUP_START[cup]
    yaw = YAW[arm]
    return [
        _w(2.0 if first else 2.5, x, y, HOVER, OPEN, f"over {cup}", yaw),
        _w(1.0, x, y, PICK_Z + 0.030, OPEN, "down to just above the grip height", yaw),
        _w(1.5, x, y, PICK_Z, OPEN, "slowly down beside the cup", yaw),
        _w(0.5, x, y, PICK_Z, OPEN, "settle", yaw),
        _w(1.5, x, y, PICK_Z, GRIP, "grip the cup", yaw),
        _w(0.5, x, y, PICK_Z, GRIP, "settle the grip", yaw),
        _w(2.0, x, y, CARRY_Z, GRIP, "lift it straight up", yaw),
    ]


def place_moves(arm, cup):
    px, py = L.POCKET_XY[L.ASSIGN[cup]]
    yaw = YAW[arm]
    side = 1.0 if arm == "left" else -1.0
    ax, ay = px, side * 0.24
    return [
        _w(3.5, px, py, CARRY_Z, GRIP, "carry it over the pocket", yaw),
        _w(0.5, px, py, CARRY_Z, GRIP, "settle", yaw),
        _w(1.5, px, py, ABOVE_MOUTH, GRIP, "down to just above the mouth", yaw),
        _w(3.0, px, py, SEAT_Z - PRESS, GRIP, "slowly down the funnel onto the floor", yaw),
        _w(0.5, px, py, SEAT_Z - PRESS, GRIP, "settle", yaw),
        _w(1.5, px, py, SEAT_Z - PRESS, OPEN, "let go of the cup", yaw),
        _w(1.5, px, py, HOVER, OPEN, "straight up", yaw),
        _w(2.0, ax, ay, HOVER, OPEN, "aside, out of the other arm's way", yaw),
    ]


def _hold(plan, seconds, label):
    """Wait where the plan ends."""
    last = plan[-1]
    x, y, z = last.pos
    return Waypoint(seconds, (x, y, z), last.finger, label, last.yaw)


def plans() -> dict:
    """Both arms' plans. The arms place by turns (their TCPs must stay 20 cm apart): left, right, left, right. Each
    arm carries on to its next cup while the other places, and waits, holding its cup, until the carrier is free."""
    plan = {"left": [], "right": []}
    order = [("left", "cup_a"), ("right", "cup_c"), ("left", "cup_b"), ("right", "cup_d")]
    free_at = 0.0  # when the carrier is free for the next placement
    first = {"left": True, "right": True}
    for arm, cup in order:
        pick = pick_moves(arm, cup, first[arm])
        first[arm] = False
        plan[arm] += pick
        ready = _duration(plan[arm])
        start = max(ready, free_at - OVERLAP)
        if start > ready:
            plan[arm].append(_hold(plan[arm], start - ready, "wait, holding the cup, until the carrier is free"))
        place = place_moves(arm, cup)
        plan[arm] += place
        free_at = start + _duration(place)
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
