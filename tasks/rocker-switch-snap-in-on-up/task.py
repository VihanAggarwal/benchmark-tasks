"""On the station table, real size: a grey steel control panel (100 x 160 x 1.5 mm, one 22 x 30 mm cutout) lies flat on
two 40 mm steel rails, its far (+x) edge carrying an orange band: that is the panel's top edge. On a foam pad in front
of it stands a KCD1-type rocker switch (body 28 x 20.5 mm, flange 31 x 25 mm, 12 g), rocker up, its printed I half
toward the robot: wrong way round for this panel.

Goal: the switch pressed into the cutout until its flange lies on the plate (within 0.5 mm), square to it (within 3
degrees), its I half toward the orange band (within 20 degrees), both grippers 30 mm clear and the switch still for 1 s.
Turned half a turn (I half toward the robot) it is half right (0.5). Left proud by up to 3 mm, or crooked, a quarter.
It has to be picked up by its rocker, turned half a turn about the vertical in the hand and lowered into a cutout with
0.75 mm of play along x and 1.0 mm along y.

The switch and the panel are mechkit compounds (mechkit.py, next to this file). No declared software rule: the spring
tabs of the real part are not modelled (see README), the switch seats under its own weight and the pads' push. The arms
are read through rigstate.py (the let-go check).

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
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import SWITCH  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Press the rocker switch into the panel's cutout until it sits flush, with its I mark toward the "
               "panel's top edge (the orange band), and let go.")


# -- the parts --------------------------------------------------------------------------------------------------------
def switch_shapes():
    """The switch in its own frame: origin at the centre of the flange's rear face, z toward the rocker, x toward
    the I half. Body with its rear edges eased (a lead-in), flange, rocker, the printed I and O."""
    ch, d = L.CHAMFER, L.DEPTH
    t = 0.0015  # the thickness of an easing slab
    c45 = math.radians(45.0)
    s2 = math.sqrt(0.5)
    out = [
        mk.box((0.0, 0.0, -(d - ch) / 2), (L.BODY_X, L.BODY_Y, d - ch), L.SW_BLACK),
        mk.box((0.0, 0.0, -d + ch / 2), (L.BODY_X - 2 * ch, L.BODY_Y - 2 * ch, ch), L.SW_BLACK),
        mk.box((0.0, 0.0, L.FL_T / 2), (L.FL_X, L.FL_Y, L.FL_T), L.SW_FLANGE),
        mk.box((0.0, 0.0, L.FL_T + L.RK_H / 2), (L.RK_X, L.RK_Y, L.RK_H), L.SW_BLACK),
    ]
    run = ch * math.sqrt(2.0)
    # each easing slab lies on the 45 degree face, its inner side t below it (it stays inside the body)
    for sgn in (1, -1):
        mid_y = sgn * (L.BODY_Y / 2 - ch / 2)
        mid_z = -d + ch / 2
        out.append(mk.box((0.0, mid_y - sgn * s2 * t / 2, mid_z + s2 * t / 2),
                          (L.BODY_X - 2 * ch, run, t), L.SW_BLACK, ypr=(0.0, 0.0, sgn * c45)))
        mid_x = sgn * (L.BODY_X / 2 - ch / 2)
        out.append(mk.box((mid_x - sgn * s2 * t / 2, 0.0, mid_z + s2 * t / 2),
                          (run, L.BODY_Y - 2 * ch, t), L.SW_BLACK, ypr=(0.0, -sgn * c45, 0.0)))
    zt = L.TOP_OF_ROCKER + L.MARK_T / 2
    out.append(mk.box((0.0045, 0.0, zt), (0.006, 0.0016, L.MARK_T), L.SW_WHITE))  # I, on the +x half
    out.append(mk.box((-0.0045, 0.0, zt), (0.006, 0.006, L.MARK_T), L.SW_WHITE))  # O: a white square...
    out.append(mk.box((-0.0045, 0.0, zt + L.MARK_T), (0.0034, 0.0034, L.MARK_T), L.SW_BLACK))  # ...with a black eye
    return out


def panel_shapes():
    plate_z = L.RAIL_H + L.PL_T / 2
    out = list(mk.plate_with_hole((0.0, 0.0, plate_z), (L.PL_X, L.PL_Y), (L.HOLE_X, L.HOLE_Y), L.PL_T, color=L.STEEL))
    rx = L.PL_X / 2 - L.RAIL_X / 2
    for sgn in (-1, 1):
        out.append(mk.box((sgn * rx, 0.0, L.RAIL_H / 2), (L.RAIL_X, L.PL_Y, L.RAIL_H), L.RAIL_STEEL))
    out.append(mk.box(((L.BAND_X0 + L.BAND_X1) / 2, 0.0, L.PLATE_TOP + L.BAND_T / 2),
                      (L.BAND_X1 - L.BAND_X0, L.PL_Y, L.BAND_T), L.ORANGE))
    return out


SWITCH_PART = mk.assembly(SWITCH, origin=(L.SW_X0, L.SW_Y0, L.TOP + L.SW_Z0), mass=L.SW_MASS, yaw=L.SW_YAW0,
                          shapes=switch_shapes(), color=L.SW_BLACK, friction=L.SW_FRICTION, tight_fit=True)
PANEL_PART = mk.assembly("panel", origin=(L.PX, L.PY, L.TOP), shapes=panel_shapes(), color=L.STEEL,
                         friction=L.PANEL_FRICTION, tight_fit=True)
FOAM = Box("foam", pos=(L.FOAM_X, L.FOAM_Y, L.TOP + L.FOAM_H / 2), size=(L.FOAM_SX, L.FOAM_SY, L.FOAM_H),
           color=L.FOAM_GREY)
OBJECTS = [PANEL_PART, FOAM, SWITCH_PART]


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it onto the highest box under it; it has no cutout. Offline
# only (never in the simulator), this model corrects the switch's height over the panel: a switch whose centre is within
# the cutout's capture range (play plus the 2 mm easing) and turned within 3 degrees of square seats flange down once
# the hand lets go (or is pressed to that height by the hand); anything else rests on the plate on its rear face.
HOLD_OPENING = 0.4  # a gripper closed this far or less, with its pads on the rocker, is holding


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seated = False

    @staticmethod
    def holding(arms, origin) -> bool:
        read = rigstate.read_arms(arms)
        if read is None:
            return False
        rocker_z = origin[2] + L.FL_T + L.RK_H / 2
        for tcp, tips, opening in read:
            if (opening < HOLD_OPENING and abs(float(tcp[0]) - origin[0]) < 0.02 and abs(float(tcp[1]) - origin[1]) < 0.02
                    and abs(float(tips[2]) - rocker_z) < 0.03):
                return True
        return False

    def states(self, objects, arms) -> dict:
        sw = objects[SWITCH]
        o = np.asarray(sw.origin, dtype=float).copy()
        q = np.asarray(sw.quat_xyzw, dtype=float)
        r = S.rotation(q)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
        err = abs(math.remainder(yaw, math.pi))  # the cutout is the same turned half a turn
        err = min(err, math.pi - err)
        dx, dy = float(o[0] - L.PX), float(o[1] - L.PY)
        over = abs(dx) <= L.PL_X / 2 and abs(dy) <= L.PL_Y / 2
        fits = abs(dx) <= L.CAPTURE_X and abs(dy) <= L.CAPTURE_Y and math.degrees(err) <= L.FIT_YAW_DEG
        if over:
            held = self.holding(arms, o - np.array([0.0, 0.0, L.TOP]))
            z_seat = L.TOP + L.SEAT_Z
            z_rest = L.TOP + L.PLATE_TOP + L.DEPTH  # standing on its rear face on the plate
            if fits:
                if self.seated or (not held and o[2] < z_rest + 0.0005):
                    self.seated = True  # let go over the cutout: it drops flange down and stays
                    o[2] = z_seat
                o[2] = max(o[2], z_seat)
                if o[2] < z_rest:  # in the cutout: the play bounds its offset
                    o[0] = L.PX + max(-L.CLEAR_X, min(L.CLEAR_X, dx))
                    o[1] = L.PY + max(-L.CLEAR_Y, min(L.CLEAR_Y, dy))
            else:
                o[2] = max(o[2], z_rest)
        elif not over:
            self.seated = False
        return {SWITCH: mk.state_from_poses(SWITCH_PART, (*o, *q), source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(SWITCH), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, gap mm, tilt deg, x from cutout mm, y from cutout mm]


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the switch's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """In the cutout flange down, square, the I half toward the top edge, both grippers 30 mm clear and the switch
    still for 1 s, no latched fall; held 1 s. This runs first every control step, so it also feeds the tracker (and,
    without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {SWITCH: objects.get(SWITCH)}
    arms = rigstate.arms()
    if no_simulator(parts):
        parts = _OFFLINE.states(parts, arms)
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
    row["source"] = getattr(parts.get(SWITCH), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["gap_mm"], d["tilt_deg"], d["from_cutout_mm"][0], d["from_cutout_mm"][1]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the switch fell off the table, or unless it is let go and still, or not in the cutout; 1.0 flange within
    0.5 mm of the plate, square within 3 degrees, I half toward the top edge; 0.5 the same turned half a turn; 0.25
    in the cutout proud by up to 3 mm or crooked up to 8 degrees; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
OPEN_SW = -(L.RK_X + 0.024) / 2  # 42 mm round the 18 mm rocker: 12 mm each side
GRIP_SW = -(L.RK_X - 0.008) / 2  # 10 mm commanded across the rocker's 18 mm
YAW_PICK, YAW_PLACE = -math.pi / 2, math.pi / 2  # both close along x; half a turn between them

GRIP_TIPS = L.SW_Z0 + L.FL_T + 0.0005  # pad tips 0.5 mm over the flange's top, either side of the rocker
GRIP_TCP = GRIP_TIPS + PAD_TIP
HOVER = 0.115  # carry height: the body's rear face is 88 mm clear of the foam, 33 mm over the plate
ABOVE_NOSE = 0.004  # the body's rear face 4 mm over the plate's top before it goes in
ABOVE_TIPS = L.PLATE_TOP + L.DEPTH + ABOVE_NOSE + L.FL_T + 0.0005
ABOVE_TCP = ABOVE_TIPS + PAD_TIP
SEAT_TIPS = L.PLATE_TOP + L.FL_T + 0.0005  # pad tips 0.5 mm over the flange's top when it lies on the plate
PRESS = 0.002  # commanded 2 mm lower: the pads push on the flange
SEAT_TCP = SEAT_TIPS + PAD_TIP - PRESS
PARK = (0.22, 0.34, 0.160)


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def pick_moves():
    x, y = L.SW_X0, L.SW_Y0
    return [
        _w(2.0, x, y, HOVER, OPEN_SW, "over the switch", YAW_PICK),
        _w(1.2, x, y, GRIP_TCP + 0.020, OPEN_SW, "down to just above its rocker", YAW_PICK),
        _w(1.5, x, y, GRIP_TCP, OPEN_SW, "slowly down: pads beside the rocker", YAW_PICK),
        _w(0.6, x, y, GRIP_TCP, OPEN_SW, "settle", YAW_PICK),
        _w(1.5, x, y, GRIP_TCP, GRIP_SW, "grip the rocker across its depth", YAW_PICK),
        _w(1.5, x, y, HOVER, GRIP_SW, "lift it off the foam", YAW_PICK),
    ]


def place_moves(dy=0.0, turn=True):
    x, y = L.PX, L.PY + dy
    yaw = YAW_PLACE if turn else YAW_PICK
    return [
        _w(3.0, x, y, HOVER, GRIP_SW, "carry it over the cutout, turning it half a turn", yaw),
        _w(0.8, x, y, HOVER, GRIP_SW, "settle", yaw),
        _w(1.5, x, y, ABOVE_TCP, GRIP_SW, "down until the body's rear face is 4 mm over the plate", yaw),
        _w(0.8, x, y, ABOVE_TCP, GRIP_SW, "settle", yaw),
        _w(3.0, x, y, SEAT_TCP, GRIP_SW, "press it slowly into the cutout until the flange lies on the plate", yaw),
        _w(0.8, x, y, SEAT_TCP, GRIP_SW, "settle", yaw),
        _w(1.5, x, y, SEAT_TCP, OPEN, "let go", yaw),
        _w(1.5, x, y, HOVER, OPEN, "straight up", yaw),
        _w(2.0, *PARK[:2], PARK[2], OPEN, "aside", yaw),
    ]


# Wrong ways (counterfactuals): not turned (I half toward the robot), or put down 6 mm off the cutout.
VARIANTS = ("unturned", "offset")


def plans(variant: str = "") -> dict:
    """The left arm's plan for the demo or a counterfactual variant: grip the rocker, lift, turn and carry over the
    cutout, lower, press until the flange lies on the plate, let go, rise and move aside. The right arm has nothing to
    do (it holds the ready pose)."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    left = pick_moves() + place_moves(dy=0.006 if variant == "offset" else 0.0, turn=variant != "unturned")
    return {"left": left}


def demo():
    """DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      unturned   placed without the half turn: the I half toward the robot (0.5)
      offset     put down 6 mm off the cutout's centre line: it rests on the plate (0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
