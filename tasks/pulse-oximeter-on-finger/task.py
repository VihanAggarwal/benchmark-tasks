"""On the station table, real size: a fixed manikin right hand and forearm (rigid skin-tone plastic) lying on its
little-finger edge with the thumb up, its straight index finger (15 mm across, rounded tip) pointing at the robot on
the midline, tip at x 0.270, axis 75 mm above the table, nail toward +y. A loose fingertip pulse oximeter (a Contec
CMS50D: 57 x 31 x 32 mm, 50 g with its cells; blue) lies on its side (pivot upright) on the midline, centre at x
0.180, lips toward the fingertip and 61.5 mm short of it, display shell toward +y.

The clip is shut at rest (mouth 6 mm, the finger is 15 mm). Its two shells turn on one pivot 20 mm from the back end,
closed by a torsion spring (0.15 N m/rad from shut). Pinching the two black grip pads on the 20 mm tail opens it, to
28 degrees with the tails together (mouth 23.7 mm); a grasp ahead of the pivot presses the shells together and holds
it shut. Opened, it has to be carried over the fingertip (4.4 mm clear a side at the lips, 1.1 mm at the cushions'
inner end) until the fingertip meets the stop inside, then let go: the cushions close on the finger at 16.3 degrees
and hold the unit by the spring. See scoring.py for the rules and the tiers.

The manikin and the oximeter are mechkit assemblies (mechkit.py next to this file); the arms are read through
rigstate.py. Neither changes what the embodiment does. There is no rule or weld: the pinch, the entry, the stop and
the hold are contacts and one sprung joint.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the demo uses the right arm; DEMO_IDLE_ARM=right runs the same plan with the left arm instead.
Settle:  DEMO_HOLD_ONLY=1 gives no plan (both arms hold the ready pose).
Variants (the README's counterfactuals): DEMO_VARIANT=body | pivot | light | part | just_open | early_release |
         no_release | hover.
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built as compound bodies)
import rigstate  # noqa: E402
from offline_clip import OfflineClip  # noqa: E402
from scoring import JAW, MANIKIN, OXIMETER, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 must hold for 1 s on top of the scorer's own 1 s of release and stillness, so the trial cannot end while a
# gripper holds or touches the oximeter.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Clip the blue pulse oximeter onto the manikin's outstretched index finger. Pinch the oximeter's back "
               "end to open its jaws and keep them open while you slide it over the fingertip as far as it will go "
               "(the fingertip within 2 mm of the stop inside). Then open the gripper and move it away, leaving the "
               "oximeter clipped on the finger.")

MM = 0.001


# -- the manikin hand (fixed): table coordinates, origin at the table top under the fingertip ----------------------
def _hand_shapes():
    ox = L.TIP_X
    s = L.SKIN

    def at(x, y, z):
        return (x - ox, y, z)

    finger_len = L.KNUCKLE_X - (L.TIP_X + 0.004)
    shapes = [
        mk.cylinder(at((L.TIP_X + 0.004 + L.KNUCKLE_X) / 2, 0.0, L.FINGER_Z), L.FINGER_D / 2, finger_len, "X", s),
    ]
    for x0, x1, d in L.TIP_STEPS:
        shapes.append(mk.cylinder(at((x0 + x1) / 2, 0.0, L.FINGER_Z), d / 2, x1 - x0, "X", s))
    shapes += [
        mk.box(at(0.3895, -0.0005, 0.0425), (0.095, 0.025, 0.085), s),  # back of the hand
        *[mk.cylinder(at(0.352, -0.0305, z), 0.010, 0.035, "Y", s) for z in (0.053, 0.032, 0.011)],  # curled fingers
        mk.box(at(0.3995, -0.0255, 0.0425), (0.075, 0.025, 0.085), s),  # heel of the hand
        mk.cylinder(at(0.377, -0.026, 0.076), 0.009, 0.050, "X", s),  # thumb, folded over the curled fingers
        mk.cylinder(at(0.527, -0.018, 0.029), 0.029, 0.180, "X", s),  # forearm
    ]
    return shapes


MANIKIN_SPEC = mk.assembly(MANIKIN, origin=(L.TIP_X, 0.0, L.TOP), shapes=_hand_shapes(), color=L.SKIN,
                           friction=L.HAND_FRICTION, tight_fit=True)


# -- the oximeter (loose): base = battery shell, link "jaw" = display shell ------------------------------------------
def _box(u, y, z, color):
    """A box from (u0, u1), (Y0, Y1), (Z0, Z1) in mm (u from the back end) in the oximeter's base frame."""
    (u0, u1), (y0, y1), (z0, z1) = u, y, z
    centre = (((u0 + u1) / 2) * MM - L.U0, ((y0 + y1) / 2) * MM, ((z0 + z1) / 2) * MM)
    return mk.box(centre, ((u1 - u0) * MM, (y1 - y0) * MM, (z1 - z0) * MM), color)


def _battery_shapes():
    out = [_box(u, y, z, c) for _, u, y, z, c in L.BATTERY_SHELL]
    out += [_box(u, (-13.0, -face), L.BASE_CUSHION_Z, L.CUSHION) for u, face in zip(L.CUSHION_STEPS_U, L.CUSHION_FACES)]
    return out


def _display_shapes():
    out = [_box(u, y, z, c) for _, u, y, z, c in L.DISPLAY_SHELL]
    out += [_box(u, (face, 13.0), L.JAW_CUSHION_Z, L.CUSHION) for u, face in zip(L.CUSHION_STEPS_U, L.CUSHION_FACES)]
    return out


JAW_LINK = mk.hinge(JAW, "Z", (L.PIVOT_X, 0.0, 0.0), _display_shapes(), lower=0.0, upper=L.OPEN_LIMIT,
                    mass=L.JAW_MASS, spring=L.SPRING, damping=L.DAMPING, friction=0.0, color=L.BLUE)
OXIMETER_SPEC = mk.assembly(OXIMETER, origin=(L.START[0], L.START[1], L.TOP + L.START[2]), mass=L.BASE_MASS,
                            shapes=_battery_shapes(), joints=[JAW_LINK], color=L.BLUE, friction=L.UNIT_FRICTION,
                            tight_fit=True)

OBJECTS = [MANIKIN_SPEC, OXIMETER_SPEC]

# -- the scorer and its hooks -----------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineClip()  # only stepped without a simulator (see offline_clip.py)


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    OFFLINE.reset()


@mk.on_reset
def _mechkit_reset() -> None:
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the clip: mechkit has no environment AND the states handed in were not
    read from one (mechkit marks those ``source="sim"``). Either sign of a simulator keeps its states untouched."""
    return mk.env() is None and getattr(objects.get(OXIMETER), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    """Without a simulator nothing opens the clip (the offline checker only carries gripped boxes), so
    offline_clip.py moves it from the arms' motion; the scorer then sees those poses."""
    OFFLINE.step(arms)
    origin, yaw, q = OFFLINE.pose()
    out = dict(objects)
    out[OXIMETER] = mk.state_from_poses(OXIMETER_SPEC, (*origin, *mk.yaw_quat(yaw)), np.zeros(6), joints={JAW: q},
                                        source="offline")
    return out


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
    """Clipped on at the stop, slid on with the jaws open, let go and still for 1 s; held 1 s. This runs every
    control step, so it also feeds the tracker (and, without a simulator, the offline clip kinematics)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    offline = no_simulator(objects)
    if offline:
        objects = offline_states(objects, arms)
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    if offline:
        row["offline"] = {"mode": OFFLINE.mode, "holder": OFFLINE.holder, "overtravel_mm": round(OFFLINE.overtravel, 2)}
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 clipped on within 2 mm of the stop, slid on with the jaws open; 0.75 clipped on otherwise; 0.5 pushed
    fully on but held; 0.25 pinched open in the air; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -------------------------------------------------------------------------------------------------
# Open loop: fixed waypoints from the setup text, no simulator state read. One arm on the midline, yaw 0 throughout
# (fingers close along y). Finger targets: the joint value is minus half the commanded gap.
#
# The pick: the TCP 0.5 mm behind the back end (x 0.1510), in the middle of the 19.9 mm window in which the gripper
# pads bear on the grip pads and not on the body ahead of the pivot. Squeezed to a 25 mm command (15 N a finger, 5.2 N
# of it against the spring, the rest on the open stop), each shell turns about its grip pad's front edge (10.5 mm
# ahead of the TCP), so the pivot ends 24.7 mm ahead of the TCP, the stop's corners 31.0 mm and the lips 59.9 mm.
# Carried at the finger's height, the fingertip passes the lips at TCP x 0.2101 and meets the stop at 0.2390; the
# command goes on 5.5 mm more, loading the soft wrist (about 3 N) so that letting go cannot draw the unit off the stop
# (closing the jaws shortens the pad's lever by 1.7 mm). Then the gripper opens to 70 mm, rises, and moves back.
PICK_X = 0.1510
CARRY_Z = 0.0755  # the TCP with the unit's centre at the finger's height (the unit sits 0.5 mm below the TCP)
PICK_Z = 0.016  # pad tips 3 mm above the table
HOVER_Z = 0.120
APPROACH_X = 0.1995  # lips 10.6 mm short of the fingertip
SLIDE_END_X = 0.2445  # 5.5 mm past the stop contact
CLEAR_XYZ = (0.160, 0.000, 0.180)  # back over the start, every finger zone well clear of the clipped unit
OPEN = -0.030  # 60 mm apart: 11.5 mm from each grip pad
SQUEEZE = -0.0125  # a 25 mm command across the 37 mm of the two grip pads
RELEASE = -0.035  # 70 mm apart
EARLY_X = 0.2251  # early release / hover: the fingertip 15 mm inside


def _variant() -> dict:
    """The demo's knobs, and the README's counterfactual runs (each the demo with one change)."""
    v = os.environ.get("DEMO_VARIANT", "")
    k = {"pick_x": PICK_X, "squeeze": SQUEEZE, "slide_end": SLIDE_END_X, "release": True}
    if v == "body":  # grasp round the unit's centre (forward waypoints shifted by the same 29 mm)
        k["pick_x"] = L.START[0]
    elif v == "pivot":  # grasp over the pivot (shifted 20.5 mm)
        k["pick_x"] = L.START[0] + L.PIVOT_X
    elif v == "light":
        k["squeeze"] = -0.0345 / 2
    elif v == "part":
        k["squeeze"] = -0.0320 / 2
    elif v == "just_open":
        k["squeeze"] = -0.0300 / 2
    elif v == "early_release":
        k["slide_end"] = EARLY_X
    elif v == "no_release":
        k["release"] = False
    elif v == "hover":
        k["slide_end"], k["release"] = EARLY_X, False
    elif v:
        raise ValueError(f"DEMO_VARIANT={v!r} is not one of body, pivot, light, part, just_open, early_release, "
                         f"no_release, hover")
    shift = k["pick_x"] - PICK_X
    k["approach"], k["slide_end"] = APPROACH_X + shift, k["slide_end"] + shift
    return k


def _at(x, y, h):
    return (x, y, L.TOP + h)


def plan() -> list[Waypoint]:
    """One arm's plan (23.5 s): pinch the tail, lift, slide over the fingertip with the jaws open, let go, clear."""
    k = _variant()
    x, sq = k["pick_x"], k["squeeze"]
    end = k["slide_end"]
    slide_s = max(1.0, (end - k["approach"]) / 0.010)  # 10 mm/s
    wps = [
        Waypoint(2.0, _at(x, 0.0, HOVER_Z), OPEN, "over the tail, fingers 60 mm apart", 0.0),
        Waypoint(1.0, _at(x, 0.0, PICK_Z + 0.030), OPEN, "down to 3 cm above the pick, pad tips above the unit", 0.0),
        Waypoint(1.5, _at(x, 0.0, PICK_Z), OPEN, "slowly down either side of the grip pads", 0.0),
        Waypoint(0.5, _at(x, 0.0, PICK_Z), OPEN, "settle before the pinch", 0.0),
        Waypoint(1.5, _at(x, 0.0, PICK_Z), sq, "pinch the tail: the jaws open to the stop", 0.0),
        Waypoint(0.5, _at(x, 0.0, PICK_Z), sq, "settle the pinch", 0.0),
        Waypoint(1.5, _at(x, 0.0, CARRY_Z), sq, "lift: the unit's centre at the finger's height", 0.0),
        Waypoint(2.0, _at(k["approach"], 0.0, CARRY_Z), sq, "forward: the lips 10.6 mm short of the fingertip", 0.0),
        Waypoint(1.0, _at(k["approach"], 0.0, CARRY_Z), sq, "settle: the wrist catches up before the slide", 0.0),
        Waypoint(slide_s, _at(end, 0.0, CARRY_Z), sq, "slide over the fingertip to the stop, jaws open", 0.0),
        Waypoint(0.5, _at(end, 0.0, CARRY_Z), sq, "settle on the stop", 0.0),
    ]
    if k["release"]:
        wps += [
            Waypoint(1.5, _at(end, 0.0, CARRY_Z), RELEASE, "open: the cushions close on the finger", 0.0),
            Waypoint(0.5, _at(end, 0.0, CARRY_Z), RELEASE, "settle", 0.0),
            Waypoint(1.5, _at(end, 0.0, 0.160), RELEASE, "straight up, clear of the grip pads", 0.0),
            Waypoint(1.5, _at(*CLEAR_XYZ[:2], CLEAR_XYZ[2]), RELEASE, "back, away from the oximeter", 0.0),
            Waypoint(2.0, _at(*CLEAR_XYZ[:2], CLEAR_XYZ[2]), RELEASE, "hold clear", 0.0),
        ]
    else:
        wps.append(Waypoint(6.0, _at(end, 0.0, CARRY_Z), sq, "hold it there, still pinched", 0.0))
    return wps


def demo():
    """One arm (the right). DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=right: the left arm runs the same plan (on the midline with yaw 0 the plan is its own mirror);
    DEMO_IDLE_ARM=left: the right arm's plan as usual. DEMO_VARIANT: see _variant()."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    arm = "left" if os.environ.get("DEMO_IDLE_ARM") == "right" else "right"
    return {arm: plan()}
