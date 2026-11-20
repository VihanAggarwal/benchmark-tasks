"""On the station table, real size: a parked cobot's forearm (a 63 mm tube on two stands, its wrist, tool flange and
two-finger tool at the -y end) lying across the table in front of both arms, with its dress pack on top: an igus
triflex R TRE.40 cable hose (43 mm across, 13.9 mm links) running out of the exit end of an igus triflex RSE(C)
TR.RSE.40 linear retraction unit toward the wrist, its moving end not yet fixed. At the exit sits an igus TR.40.30
quick-release protector, open: a loose 55 mm square shell round the hose, 27 mm long, in a cradle on the forearm,
its lid stood open on its hinge. A yellow band marks one link of the hose, just inside the open protector.

Goal: the maker's pretension step: the hose pulled forward against the unit's elastic band, the protector closed and
latched on it at the exit, both hands away; the protector then rests on the exit face and holds the hose two links
(27.8 mm) out, the band just outside the protector's wrist-side face.

Physics, one rule: the hose is a link on a sprung slide (about 4 N two links out; an estimate) and runs back into
the unit when let go; the shut lid's rib, sitting in a waist, carries the protector with the hose until it meets the
exit face (form fit and contact). A lid pushed down over a crest stops about 8 deg open on its rib. The rule: within
2.5 deg of shut the quick-lock catches (an on_step drive target holds the lid on its shut stop); unlatched, its
spring opens it. The scorer reads joints and the arms only.

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
import offline_dresspack  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import RING, ROBOT  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # on top of the scorer's own 1 s window

INSTRUCTION = ("Pretension the dress-pack hose two links and latch its protector on it at the retraction unit's "
               "exit, then let go of both and move the grippers clear.")

# -- the parts --------------------------------------------------------------------------------------------------------
ROBOT_PART = mk.assembly(ROBOT, origin=L.ORIGIN, shapes=P.robot_shapes(), joints=[P.hose_link()], color=L.GREY,
                         friction=L.HOSE_FRICTION)
RING_PART = mk.assembly(RING, origin=L.ORIGIN, shapes=P.shell_shapes(), joints=[P.lid_link()], mass=L.RING_MASS,
                        color=L.ORANGE, friction=L.RING_FRICTION)
OBJECTS = [ROBOT_PART, RING_PART]

_OFFLINE = offline_dresspack.DressPackModel(ROBOT_PART, RING_PART)
_LATCH = {"on": False}


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(ROBOT), "source", "sim") != "sim"


def latch_target(lid_q: float, latched: bool) -> tuple[bool, float]:
    """The snap latch: (latched, the lid's drive target). Latched once within LATCH_AT of shut, until reset."""
    latched = latched or (lid_q + L.LID_OPEN) <= L.LATCH_AT
    return latched, (-L.LID_OPEN - L.LATCH_HOLD) if latched else 0.0


@mk.on_step
def _latch(states) -> None:
    st = states.get(RING) if hasattr(states, "get") else None
    if st is None:
        return
    try:
        q = float(st.joints["lid"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    _LATCH["on"], target = latch_target(q, _LATCH["on"])
    mk.set_targets(RING_PART, "lid", target)


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def _reset_all() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _LATCH["on"] = False


@rigstate.on_reset
def _new_episode() -> None:
    _reset_all()


@mk.on_reset
def _kit_reset() -> None:
    _reset_all()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The hose two links out (within 5 mm), the lid shut, both hands 4 cm away and the hose still on every step of
    the last 1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the
    offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _reset_all()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "latched": _OFFLINE.latched if offline else _LATCH["on"], **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    if offline:
        row["offline"] = {"holder": _OFFLINE.holder}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 never pulled near the mark; 0.25 pulled to within 7 mm of two links; 0.5 the lid shut while it was; 0.75
    now shut with the hose within 5 mm of two links; 1.0 that, hands off and still for the last 1 s."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. The right hand takes the
# hose round a crest on the wrist side (pads closing along x, yaw +pi/2) and draws it toward the wrist past the
# waist that fits the rib, to 40.75 mm out (layout.PULL_START, the wrist's give included: layout.PULL_CMD), where a
# crest is centred under the protector's rib. It holds there while the left hand, its fingers shut (closing along y,
# yaw 0), pushes the lid over from its +x side with the fingers' -x face and then their tips, along the lid's arc
# (layout.push_tip), and presses it down: the rib stops on the crest about 8 deg open. Still pressing, the right hand
# eases the hose back toward the unit (layout.PULL_END_CMD): as the waist passes under the rib the lid drops in and
# the quick-lock catches; the waist's wall then carries the protector onto the exit face and the hose stops 27.8 mm
# out. The right hand lets go and both go. (Two GPU runs that stopped the hose on the waist directly missed its
# 4 mm window: see layout.WRIST_C.)
YAW_L, YAW_R = 0.0, math.pi / 2


def fingers(gap: float) -> float:
    return -gap / 2


F_OPEN_R = fingers(0.060)
F_GRIP = fingers(L.HOSE_GRIP_GAP)
F_SHUT = fingers(0.0)
PARK_L = (0.200, 0.240, 0.200)
PARK_R = (0.220, -0.250, 0.200)


def _w(x, y, z):
    """World point from a base-frame point."""
    return (L.ARM_X + x, y, L.TOP + z)


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


def right_moves(pull: float = L.PULL_CMD) -> list:
    y0 = L.grip_y()
    z = L.HOSE_Z
    y1 = y0 - pull
    return [
        Waypoint(2.5, _w(0.0, y0, z + 0.060), F_OPEN_R, "over the hose's crest, fingers 60 mm apart", YAW_R),
        Waypoint(2.0, _w(0.0, y0, z), F_OPEN_R, "down either side of the crest", YAW_R),
        Waypoint(0.5, _w(0.0, y0, z), F_OPEN_R, "settle", YAW_R),
        Waypoint(1.5, _w(0.0, y0, z), F_GRIP, "grip the crest", YAW_R),
        Waypoint(0.5, _w(0.0, y0, z), F_GRIP, "settle the grip", YAW_R),
        Waypoint(5.0, _w(0.0, y1, z), F_GRIP, "draw the hose toward the wrist against the unit's spring", YAW_R),
        Waypoint(1.0, _w(0.0, y1, z), F_GRIP, "hold it there: a crest under the protector's rib", YAW_R),
    ]


def lid_path(into: float = 0.002) -> list:
    """The left pad-tip point's base-frame (x, z) along the lid's arc, from open to pressed shut."""
    pts = []
    for deg in list(range(95, -1, -5)):
        pts.append((deg, L.push_tip(math.radians(deg), into=into)))
    return pts


def left_moves(start: float, release_right: float, shut_extra: float = 0.010, variant: str = "") -> list:
    # shut_extra: commanded past the shut face (on top of lid_path's 2 mm), 12 mm in all: the lid's spring pushes back
    # about 4 N at shut and the arm sags under a press (GPU lessons: command a press 8 to 10 mm past its travel). In
    # both 20 Nov GPU runs the lid stopped 8.6 to 9 deg open: the rib on a crest (the hose 25.2 and 31.3 mm out, off
    # the waist), not a short press.
    """Wait, then come in on the lid's +x side (while the hose is still being drawn) and push it over along its arc
    once it is out, press it down, keep pressing while the hose is eased back and let go, lift away."""
    path = lid_path()
    deg0, (x0, z0) = path[0]
    nx, nz = L.lid_normal(math.radians(deg0))
    back = 0.020
    plan = [Waypoint(3.0, PARK_L[:2] + (L.TOP + PARK_L[2],), F_SHUT, "fingers shut, aside", YAW_L)]
    _hold(plan, start - 3.0, "wait for the hose to be drawn")
    plan += [Waypoint(1.5, _w(x0 + back * nx, L.RING_Y, z0 + back * nz + L.PAD_TIP + 0.010), F_SHUT,
                      "over beside the stood-up lid, on its far side", YAW_L),
             Waypoint(1.5, _w(x0 + back * nx, L.RING_Y, z0 + back * nz + L.PAD_TIP), F_SHUT,
                      "down beside the lid's outer face", YAW_L)]
    for deg, (x, z) in path:
        plan.append(Waypoint(0.5 if deg > 20 else 0.6, _w(x, L.RING_Y, z + L.PAD_TIP), F_SHUT,
                             f"push the lid over: {deg} deg open", YAW_L))
    x, z = path[-1][1]
    plan += [
        Waypoint(1.5, _w(x, L.RING_Y, z - shut_extra + L.PAD_TIP), F_SHUT,
                 "press it down: the rib on the crest", YAW_L),
        Waypoint(1.0, _w(x, L.RING_Y, z - shut_extra + L.PAD_TIP), F_SHUT, "keep pressing", YAW_L),
    ]
    _hold(plan, release_right + 2.5, "keep pressing while the hose is eased back and let go")
    plan += [
        Waypoint(1.5, _w(x, L.RING_Y, z + 0.050 + L.PAD_TIP), F_SHUT, "straight up off the lid", YAW_L),
        Waypoint(2.0, PARK_L[:2] + (L.TOP + PARK_L[2],), F_SHUT, "aside", YAW_L),
    ]
    return plan


def right_finish(plan, release_at: float, ease=None) -> list:
    """``ease``: (start time, how far out to command) to ease the hose back over 4 s before letting go, or None."""
    if ease is not None:
        at, back_to = ease
        _hold(plan, at, "hold the hose while the lid is pushed over and pressed onto the crest")
        last = plan[-1].pos
        plan.append(Waypoint(4.0, (last[0], L.grip_y() - back_to, last[2]), F_GRIP,
                             "ease the hose back: the rib drops into the waist, the protector onto the exit", YAW_R))
    _hold(plan, release_at, "hold the hose while the lid is closed")
    last = plan[-1].pos
    plan += [
        Waypoint(1.5, last, F_OPEN_R, "let go of the hose", YAW_R),
        Waypoint(0.5, last, F_OPEN_R, "settle", YAW_R),
        Waypoint(1.5, (last[0], -0.205, last[2] + 0.010), F_OPEN_R,
                 "slide off past the hose's free end, fingers either side of it", YAW_R),
        Waypoint(1.5, (last[0], -0.205, last[2] + 0.060), F_OPEN_R, "up, clear of the hose", YAW_R),
        Waypoint(2.0, PARK_R[:2] + (L.TOP + PARK_R[2],), F_OPEN_R, "aside", YAW_R),
    ]
    return plan


# Wrong ways (counterfactuals; the offline model predicts them):
#   onelink   held a link short (19.9 mm), the lid latched there: the hose ends one link out: 0
#   early     the right hand lets go before the lid is shut: the hose runs back; the lid cannot shut: 0.25
VARIANTS = ("onelink", "early")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    gain = 1.0 + L.WRIST_C * L.PULL_K
    right = right_moves((L.CLOSE_AT - L.PITCH) * gain if variant == "onelink" else L.PULL_CMD)
    out_at = _until(right, "hold it there")
    left_probe = left_moves(out_at, 0.0)
    shut_at = _until(left_probe, "keep pressing")
    if variant == "early":
        release, ease = out_at + 1.0, None
    elif variant == "onelink":  # held a link short (19.9 mm, a waist under the rib) and shut there
        release, ease = shut_at + 0.5, None
    else:
        ease = (shut_at, L.PULL_END_CMD)
        release = shut_at + 4.0 + 1.0
    right = right_finish(right, release, ease)
    left = left_moves(out_at, release)
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check). DEMO_VARIANT: see VARIANTS."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
