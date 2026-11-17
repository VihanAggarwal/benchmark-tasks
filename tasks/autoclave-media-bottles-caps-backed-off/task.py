"""On the station table, real size: a polypropylene autoclave pan (380 x 240 x 75 mm) stands at x 170 to 410 mm with
the front of a bench-top autoclave (door open) behind it. On the bench beside the pan, one on each side, stand two
250 mL DURAN-type media bottles (70 mm across, 143 mm tall, 150 mL of water, 0.33 kg), each with a blue GL45 screw
cap closed finger tight (0.25 N m to turn, about six times what the bottle's base friction holds). A black line runs
down each shoulder and up each cap; the two lines meet when the cap is tight, so the angle between them reads how far
the cap is backed off.

Goal: both bottles upright in the pan with their caps still on their threads but backed off from tight by a quarter
turn to one turn (90 to 360 degrees), both grippers clear and every part still. A bottle loaded with its cap tight is
a sealed pressure vessel in the cycle: a 0 for the trial. As at the bench, a cap only turns on its bottle while a hand
holds the bottle: one hand holds, the other turns.

Each bottle is a mechkit articulation: the glass body (base), the cap's lift on its thread (a slider "rise") and the
cap's turn (a hinge "turn" about the neck's axis, with the joint friction of a finger-tight cap). One declared software
rule (``_thread_rule``, a mechkit on_step callback): the slider is driven every control step to the lift the cap's
turn makes on its thread (6 mm a turn), so the cap rises as it unscrews. The scorer reads the bottles' poses and the
cap joints, and the arms through rigstate.py (the let-go check).

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
from scoring import PARTS  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

import tiltgrip  # noqa: E402  (side grips: wraps yam_arena.demo.grip_rotation)

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Stand the two bottles of growth media in the autoclave pan, ready to go into the autoclave on the "
               "next liquid cycle. Leave both bottles upright in the pan with their caps on.")


# -- the parts --------------------------------------------------------------------------------------------------------
def pan_shapes():
    return list(mk.open_box((0.0, 0.0, L.PAN_H / 2), (L.PAN_OX, L.PAN_OY, L.PAN_H), L.PAN_WALL, L.PAN_FLOOR,
                            L.PAN_WHITE))


def autoclave_shapes():
    """The front section of a bench-top autoclave, base frame at the centre of its footprint on the table."""
    d, w, h = L.AUTO_X1 - L.AUTO_X0, L.AUTO_Y, L.AUTO_H
    z0 = L.CHAMBER_Z0
    z1 = h - L.AUTO_WALL
    side = (w - L.CHAMBER_W) / 2
    return [
        mk.box((0.0, 0.0, z0 / 2), (d, w, z0), L.AUTO_STEEL),  # the chamber floor and the foot below it
        mk.box((0.0, 0.0, (z1 + h) / 2), (d, w, h - z1), L.AUTO_GREY),  # the roof
        mk.box((0.0, (L.CHAMBER_W + side) / 2, (z0 + z1) / 2), (d, side, z1 - z0), L.AUTO_GREY),  # left wall
        mk.box((0.0, -(L.CHAMBER_W + side) / 2, (z0 + z1) / 2), (d, side, z1 - z0), L.AUTO_GREY),  # right wall
    ]


def body_shapes():
    """The glass: straight body, shoulder up to the cap's seat, the GL45 neck, the indicator tape and the shoulder's
    black line (a 1.2 mm strip facing -x). Base frame at the centre of the base's underside."""
    return [
        mk.cylinder((0.0, 0.0, L.BODY_H / 2), L.BODY_R, L.BODY_H, "Z", L.GLASS),
        mk.cylinder((0.0, 0.0, (L.BODY_H + L.CAP_SEAT) / 2), L.SHOULDER_R, L.CAP_SEAT - L.BODY_H, "Z", L.GLASS),
        mk.cylinder((0.0, 0.0, (L.CAP_SEAT + L.NECK_TOP) / 2), L.NECK_R, L.NECK_TOP - L.CAP_SEAT, "Z", L.GLASS),
        mk.cylinder((0.0, 0.0, L.TAPE_Z0 + L.TAPE_H / 2), L.TAPE_R, L.TAPE_H, "Z", L.TAPE_COLOUR),
        mk.box((-(L.SHOULDER_R + L.MARK_THICK / 2), 0.0, (L.BODY_H + L.CAP_SEAT) / 2),
               (L.MARK_THICK, L.MARK_W, L.CAP_SEAT - L.BODY_H), L.MARK_BLACK),
    ]


def cap_shapes():
    """The screw cap (a 54 mm cylinder, 27 tall, its skirt on the shoulder when tight) and its black line."""
    zc = L.CAP_SEAT + L.CAP_H / 2
    return [
        mk.cylinder((0.0, 0.0, zc), L.CAP_R, L.CAP_H, "Z", L.CAP_BLUE),
        mk.box((-(L.CAP_R + L.MARK_THICK / 2), 0.0, zc), (L.MARK_THICK, L.MARK_W, L.CAP_H), L.MARK_BLACK),
    ]


def make_bottle(name, xy):
    rise = mk.slider("rise", "Z", [mk.cylinder((0.0, 0.0, L.NECK_TOP + 0.0005), 0.018, 0.001, "Z", L.CAP_BLUE)],
                     upper=L.RISE_MAX, lower=0.0, mass=L.SLIDER_MASS, spring=L.K_RISE, damping=L.D_RISE)
    turn = mk.hinge("turn", "Z", (0.0, 0.0, L.CAP_SEAT), cap_shapes(), upper=L.STOP_RAD, lower=-L.TIGHTEN_RAD,
                    mass=L.CAP_MASS, damping=L.D_TURN, friction=L.CAP_FRICTION, parent="rise")
    return mk.assembly(name, origin=(xy[0], xy[1], L.TOP), mass=L.BASE_MASS, shapes=body_shapes(),
                       joints=[rise, turn], color=L.GLASS, friction=L.BOTTLE_FRICTION)


PAN = mk.assembly("pan", origin=(L.PAN_CX, L.PAN_CY, L.TOP), shapes=pan_shapes(), color=L.PAN_WHITE,
                  friction=L.PAN_FRICTION)
AUTOCLAVE = mk.assembly("autoclave", origin=((L.AUTO_X0 + L.AUTO_X1) / 2, 0.0, L.TOP), shapes=autoclave_shapes(),
                        color=L.AUTO_GREY)
BOTTLE_A = make_bottle("bottle_a", L.A0)
BOTTLE_B = make_bottle("bottle_b", L.B0)
OBJECTS = [PAN, AUTOCLAVE, BOTTLE_A, BOTTLE_B]
SPECS = {"bottle_a": BOTTLE_A, "bottle_b": BOTTLE_B}


# -- the one declared rule: a cap rises as it unscrews ----------------------------------------------------------------
@mk.on_step
def _thread_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics): each cap's lift is
    driven to what its turn makes on a 6 mm thread (0 when the turn cannot be read)."""
    for name, spec in SPECS.items():
        st = states.get(name) if isinstance(states, dict) else None
        try:
            turn = st.joints.get("turn") if st is not None else None
        except AttributeError:
            turn = None
        mk.set_targets(spec, "rise", S.rise_target(turn))


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm past them along the approach. Finger targets: the joint value is minus half the commanded gap (8 mm narrower
# than the part).
#   1. Each arm carries its own bottle (a side grip round the body, the pads level across it) into the pan, lets go
#      and backs out.
#   2. Bottle a: the right hand holds its body with a side grip; the left hand grips its cap from above and turns the
#      wrist 190 degrees anticlockwise; the left lets go, then the right.
#   3. Bottle b, the other way round: the left hand holds, the right hand turns. Both park clear.
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
GRIP_B = -(2 * L.BODY_R - 0.008) / 2  # 62 mm across the 70 mm body
GRIP_C = -(2 * L.CAP_R - 0.008) / 2  # 46 mm across the 54 mm cap
YAW_BODY = {"left": -math.pi / 2, "right": math.pi / 2}  # the wrist turned back to point down, fingers along x
YAW_CAP0 = {"left": math.radians(-135.0), "right": math.radians(-55.0)}  # start of the wrist's turn (its window)
TURN_STEPS = (95.0, 95.0)  # the wrist turns 190 degrees anticlockwise in two moves (a slerp needs under 180 each)
CAP_TIPS_ABOVE_SEAT = 0.006  # pad tips this far over the cap's skirt
CARRY_BASE = 0.100  # the bottle's base this high above the table while carried: 25 mm over the pan's rim
# the carry's side grips: the approach turned this far from +x when picking the bottle up on the bench (the fingers
# closing mostly along x, clear of the pan) and when setting it down in the pan (clear of the pan's side wall)
PICK_PHI = {"left": math.radians(-50.0), "right": math.radians(50.0)}
PLACE_PHI = {"left": math.radians(-10.0), "right": math.radians(10.0)}
CARRY_Z = 0.075  # the carry grips the body this high (the wrist well over the pan's front wall when setting down)
CARRY_RISE = 0.110  # after letting go: the TCP straight up this far, the pads over the cap's top
RELEASE = -0.040  # let go of a bottle in the pan to 80 mm, not the full 94 (the fingers clear of the pan's walls)
UP = 0.170  # TCP height for moves over the bottles: the pad tips 9 mm over the caps
HOVER_START = 0.150
# the holding hand: a side grip on the body, the approach axis HOLD_DEP below horizontal, turned HOLD_PHI from +x
# toward the bottle's other side (the right hand holds bottle a coming in toward +y, the left hand bottle b toward -y)
HOLD_PHI = math.radians(35.0)
HOLD_DEP = math.radians(20.0)
HOLD_FLIP = {"left": False, "right": False}
HOLD_Z = 0.055  # the body's axis point the pads grip, over the bottle's base (the straight body ends at 100)
AXIS_PAST_TCP = 0.0  # that point is at the TCP (the pad tips 13 mm past it)
HOLD_BACK = 0.080  # the staging point: this far back from the grip point along the approach, fingers open
HIGH_BACK, HOLD_RAISE = 0.160, 0.060  # the high staging point: this far back along the approach and this far up
TURN_BACK = {"left": (0.25, 0.20, UP), "right": (0.25, -0.20, UP)}  # clear of both caps
PARK = {"left": (0.22, 0.30, 0.15), "right": (0.22, -0.30, 0.15)}
PARK_YAW = {"left": None, "right": math.radians(-37.0)}  # the right wrist turns back to its window to park
SPOT = {"left": ("bottle_a", L.A0, L.A1), "right": ("bottle_b", L.B0, L.B1)}
PRESS = 0.002  # lowered this far past the pan floor so the bottle rests on it before the fingers open


def side_approach(phi: float) -> np.ndarray:
    """A side grip's approach axis (unit): turned ``phi`` from +x and HOLD_DEP down."""
    return np.array([math.cos(HOLD_DEP) * math.cos(phi), math.cos(HOLD_DEP) * math.sin(phi), -math.sin(HOLD_DEP)])


def hold_approach(holder: str) -> np.ndarray:
    """The holding hand's approach axis (unit): toward the bottle and HOLD_DEP down."""
    return side_approach(HOLD_PHI if holder == "right" else -HOLD_PHI)


def hold_yaw(holder: str) -> float:
    """The holding hand's side grip as a waypoint yaw value (see side_yaw)."""
    return side_yaw(hold_approach(holder), HOLD_FLIP[holder])


def side_yaw(a, flip: bool = False) -> float:
    """A waypoint yaw value carrying a side grip's full rotation: the approach ``a``, the fingers closing across it,
    level (tiltgrip)."""
    c = np.array([-a[1], a[0], 0.0])
    c = c / np.linalg.norm(c)
    if flip:
        c = -c  # the wrist turned the other way round
    frm = np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])  # closing along y, approach down
    to = np.column_stack([np.cross(c, a), c, a])
    rot = to @ frm.T @ tiltgrip.rotation(0.0)
    return tiltgrip.TiltedYaw(0.0, rot)


def hold_points(holder: str, bottle_xy) -> tuple:
    """(grip TCP, staging TCP, high staging TCP) for ``holder`` holding the bottle standing in the pan at
    ``bottle_xy``, heights above the table."""
    return side_points(hold_approach(holder), bottle_xy, L.PAN_FLOOR)


def side_points(a, bottle_xy, floor: float, z: float = HOLD_Z) -> tuple:
    """(grip TCP, staging TCP, high staging TCP) of a side grip along ``a`` on the body of a bottle standing at
    ``bottle_xy`` on a surface ``floor`` above the table, ``z`` above its base; heights above the table."""
    g = np.array([bottle_xy[0], bottle_xy[1], floor + z])
    tcp = g - AXIS_PAST_TCP * a
    stage = g - HOLD_BACK * a
    return tcp, stage, g - HIGH_BACK * a + np.array([0.0, 0.0, HOLD_RAISE])


class _Track:
    """One arm's waypoints and its clock (heights above the table)."""

    def __init__(self) -> None:
        self.plan: list = []
        self.t = 0.0
        self.last = None

    def go(self, d, pos, finger, label, yaw) -> None:
        self.plan.append(Waypoint(float(d), (float(pos[0]), float(pos[1]), L.TOP + float(pos[2])), finger, label,
                                  yaw))
        self.t += float(d)
        self.last = (tuple(float(v) for v in pos), finger, yaw)

    def wait_until(self, t: float, label: str) -> None:
        if t > self.t + 1e-9:
            pos, finger, yaw = self.last
            self.go(t - self.t, pos, finger, label, yaw)


def _carry(tr: _Track, side: str) -> None:
    """Grip the bottle's body on the bench from the side, lift it, carry it into the pan, set it down, let go, back
    out and up, then turn the hand back to point down."""
    _, xy0, xy1 = SPOT[side]
    a0, a1 = side_approach(PICK_PHI[side]), side_approach(PLACE_PHI[side])
    sy0, sy = side_yaw(a0), side_yaw(a1)
    tcp0, stage0, high0 = side_points(a0, xy0, 0.0, CARRY_Z)
    tcp1 = side_points(a1, xy1, L.PAN_FLOOR, CARRY_Z)[0]
    lift = np.array([0.0, 0.0, CARRY_BASE])
    rise = np.array([0.0, 0.0, CARRY_RISE])
    place = tcp1 - np.array([0.0, 0.0, PRESS])
    tr.go(2.5, high0, OPEN, "beside the bottle, wrist on its side", sy0)
    tr.go(1.5, stage0, OPEN, "down beside its body", sy0)
    tr.go(1.5, tcp0, OPEN, "in round its body", sy0)
    tr.go(0.4, tcp0, OPEN, "settle", sy0)
    tr.go(1.6, tcp0, GRIP_B, "grip the body across its 70 mm (side grip)", sy0)
    tr.go(1.5, tcp0 + lift, GRIP_B, "lift it 100 mm", sy0)
    tr.go(2.0, tcp1 + lift, GRIP_B, "carry it over the pan", sy)
    tr.go(1.5, place, GRIP_B, "lower it onto the pan floor", sy)
    tr.go(0.4, place, GRIP_B, "settle", sy)
    tr.go(1.6, place, RELEASE, "let go of the body", sy)
    tr.go(2.0, tcp1 + rise, RELEASE, "straight up, the fingers either side of the neck and cap", sy)
    tr.go(1.5, tcp1 + rise - HOLD_BACK * a1, RELEASE, "back out over the cap's height", sy)


def _loosen(turner: _Track, tside: str, holder: _Track, hside: str, xy, twist: float, hold: bool = True,
            holder_after: float = 0.0) -> None:
    """One bottle in the pan at ``xy``: ``holder`` grips its body from the side and keeps it still; ``turner`` grips
    its cap from above and turns the wrist ``twist`` degrees anticlockwise (unscrewing), lets go and rises; then the
    holder lets go and backs out. ``holder_after``: the holder starts this many seconds after the turner (so the
    turner is out of its way). ``hold`` False: no holding hand (a counterfactual)."""
    x, y = xy
    t0 = max(turner.t, holder.t)
    turner.wait_until(t0, "wait")
    holder.wait_until(t0 + holder_after, "wait")
    yc = YAW_CAP0[tside]
    cap_z = L.PAN_FLOOR + L.CAP_SEAT + CAP_TIPS_ABOVE_SEAT + PAD_TIP
    turner.go(2.5, (x, y, UP), OPEN, "over the cap, wrist to its start yaw", yc)
    turner.go(2.0, (x, y, cap_z), OPEN, "slowly down round the cap", yc)
    turner.go(0.5, (x, y, cap_z), OPEN, "settle", yc)
    if hold:
        tcp, stage, high = hold_points(hside, xy)
        hy = hold_yaw(hside)
        holder.go(2.5, high, OPEN, "beside the bottle, wrist on its side", hy)
        holder.go(1.5, stage, OPEN, "down beside its body", hy)
        holder.go(1.5, tcp, OPEN, "in round its body", hy)
        holder.go(1.6, tcp, GRIP_B, "hold the body (side grip)", hy)
        turner.wait_until(holder.t, "wait for the other hand to hold the bottle")
    turner.go(1.6, (x, y, cap_z), GRIP_C, "grip the cap across its 54 mm", yc)
    turner.go(0.4, (x, y, cap_z), GRIP_C, "hold", yc)
    yaw, left = yc, twist
    for i, step in enumerate(TURN_STEPS):
        step = min(step, left)
        left -= step
        if step <= 0:
            break
        yaw += math.radians(step)
        turner.go(2.5, (x, y, cap_z), GRIP_C, f"turn the wrist {step:.0f} degrees (unscrew), move {i + 1}", yaw)
        turner.go(0.3, (x, y, cap_z), GRIP_C, "pause", yaw)
    turner.go(0.5, (x, y, cap_z), GRIP_C, "settle", yaw)
    turner.go(1.6, (x, y, cap_z), OPEN, "let go of the cap", yaw)
    turner.go(2.0, (x, y, UP), OPEN, "straight up", yaw)
    if hold:
        holder.wait_until(turner.t - 2.0, "hold the bottle while the cap turns")
        holder.go(1.6, tcp, OPEN, "let go of the body", hy)
        holder.go(1.5, stage, OPEN, "back out", hy)
        holder.go(1.5, high, OPEN, "up", hy)
        holder.go(2.5, TURN_BACK[hside], OPEN, "aside to its own side, the hand turned back to point down",
                  YAW_BODY[hside])


# Wrong ways (counterfactuals): the cap is not backed off enough (or at all), or turned with no hand on the bottle.
VARIANT_TWIST = {"sealed": 0.0, "short": 60.0, "nohold": sum(TURN_STEPS)}
VARIANTS = tuple(VARIANT_TWIST)


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant, on one clock."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    twist = VARIANT_TWIST.get(variant, sum(TURN_STEPS))
    hold = variant != "nohold"
    left, right = _Track(), _Track()
    _carry(left, "left")
    _carry(right, "right")
    left.go(2.5, TURN_BACK["left"], OPEN, "aside, the hand turned back to point down", YAW_BODY["left"])
    if twist > 0:
        _loosen(left, "left", right, "right", L.A1, twist, hold)
        _loosen(right, "right", left, "left", L.B1, twist, hold, holder_after=5.0)
    t = max(left.t, right.t)
    for tr, side in ((left, "left"), (right, "right")):
        tr.wait_until(t, "wait")
        yaw = PARK_YAW[side] if PARK_YAW[side] is not None else tr.last[2]
        tr.go(2.5, PARK[side], OPEN, "aside", yaw)
    return {"left": left.plan, "right": right.plan}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      sealed     both bottles carried into the pan with their caps left tight: a sealed bottle loaded (0)
      short      the wrist turned only 60 degrees: the cap still sealed (0)
      nohold     the caps turned with no hand on the bottles: the bottles turn with them (0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _yaw_at(side: str, t: float) -> float:
    """The commanded wrist yaw of ``side``'s plan (the current DEMO_VARIANT) at time ``t``: the offline model only,
    read while a hand grips a cap (top-down, where the plan's yaw is the wrist's turn)."""
    plan = _plan_for(side)
    prev = float(plan[0].yaw)
    for w in plan:
        if t < w.duration:
            s = t / w.duration
            s = s * s * (3.0 - 2.0 * s)
            return prev + s * (float(w.yaw) - prev)
        t -= w.duration
        prev = float(w.yaw)
    return prev


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped object rigidly and drops it straight down; it knows nothing of a cap turning on
# its thread. Offline only (never in the simulator), this model gives the scorer: (1) a bottle gripped by a closed
# gripper on its body (from above over its axis, or from the side with the pads across its axis) is carried with the
# TCP and set down on the pan's floor or the table when let go; (2) a closed gripper from above over a bottle's axis
# with the pad tips on its cap turns the cap by the change of the wrist's commanded yaw since the grip, but only while
# the other hand holds that bottle's body: otherwise the cap's torque (0.25 N m) turns the whole bottle, whose base
# friction holds about 0.04 N m, and the cap stays tight on it.
CLOSED_MAX = 0.80  # an opening under this is a closed grip on a part (70 mm of 94 is 0.74)
AXIS_TOL = 0.012  # the TCP (from above) or the grip point (from the side) within 12 mm of the bottle's axis
TOP_DOWN_DROP = 0.010  # the pad tips this far below the TCP: a grip from above (13 mm straight down)
DT = 1.0 / L.CONTROL_HZ
_PLAN_CACHE: dict = {}


def _plan_for(side: str):
    key = (os.environ.get("DEMO_VARIANT", ""), side)
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = plans(key[0])[side]
    return _PLAN_CACHE[key]


def support_z(x: float, y: float) -> float:
    """The height (m above the table) a bottle set down at (x, y) rests at: the pan's floor inside its walls."""
    inside = L.PAN_X0 <= x <= L.PAN_X1 and L.PAN_Y0 <= y <= L.PAN_Y1
    return L.PAN_FLOOR if inside else 0.0


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.k = 0
        self.pos = {n: np.array([S.STARTS[n][0], S.STARTS[n][1], 0.0]) for n in PARTS}
        self.turn = {n: 0.0 for n in PARTS}
        self.grip: dict = {}

    def _find(self, tcp, tips, opening):
        """What a hand grips: ("body", name), ("cap", name) or None."""
        if opening >= CLOSED_MAX:
            return None
        drop = float(tcp[2] - tips[2])
        for n in PARTS:
            p = self.pos[n]
            if drop >= TOP_DOWN_DROP:
                if float(np.hypot(tcp[0] - p[0], tcp[1] - p[1])) > AXIS_TOL:
                    continue
                rel = float(tips[2]) - L.TOP - p[2]
                if -0.002 <= rel <= 0.075:
                    return ("body", n)
                if L.CAP_SEAT - 0.004 <= rel <= L.CAP_TOP - 0.004:
                    return ("cap", n)
            else:
                ap = (np.asarray(tips, dtype=float) - np.asarray(tcp, dtype=float)) / PAD_TIP
                point = np.asarray(tcp, dtype=float) + AXIS_PAST_TCP * ap
                if float(np.hypot(point[0] - p[0], point[1] - p[1])) > AXIS_TOL:
                    continue
                rel = float(point[2]) - L.TOP - p[2]
                if 0.015 <= rel <= L.BODY_H - 0.012:
                    return ("body", n)
        return None

    def step(self, arms) -> None:
        self.k += 1
        t = self.k * DT
        read = rigstate.read_arms(arms)
        if read is None:
            return
        found = {side: self._find(tcp, tips, opening) for side, (tcp, tips, opening) in zip(rigstate.ARMS, read)}
        carried = set()
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            old = self.grip.get(side)
            if found[side] is None:
                self.grip.pop(side, None)
                continue
            kind, n = found[side]
            if kind == "body":
                hand = np.asarray(tcp, dtype=float) - [0, 0, L.TOP]
                if old is None or old["kind"] != "body" or old["name"] != n:
                    old = {"kind": "body", "name": n, "off": self.pos[n] - hand}
                self.grip[side] = old
                self.pos[n] = hand + old["off"]
                carried.add(n)
            else:
                yaw = _yaw_at(side, t)
                other_holds = any(f is not None and f == ("body", n) for s2, f in found.items() if s2 != side)
                if old is None or old["kind"] != "cap" or old["name"] != n or not other_holds:
                    old = {"kind": "cap", "name": n, "yaw": yaw, "turn": self.turn[n]}  # unheld: bottle turns too
                self.grip[side] = old
                self.turn[n] = float(min(max(old["turn"] + (yaw - old["yaw"]), -L.TIGHTEN_RAD), L.STOP_RAD))
        for n in PARTS:
            if n not in carried:
                self.pos[n][2] = support_z(float(self.pos[n][0]), float(self.pos[n][1]))

    def states(self) -> dict:
        out = {}
        for n in PARTS:
            p = self.pos[n]
            joints = {"turn": self.turn[n], "rise": S.rise_target(self.turn[n])}
            out[n] = mk.state_from_poses(SPECS[n], (float(p[0]), float(p[1]), L.TOP + float(p[2]), 0.0, 0.0, 0.0, 1.0),
                                         joints=joints, source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(PARTS[0]), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, turn a, turn b, rise a mm, rise b mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: both bottles' poses, caps and what the scorer
    saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Both bottles upright in the pan, each cap backed off 90 to 360 degrees and on its thread, both grippers 30 mm
    clear and every part still for 1 s, no latched fall or tip; held 1 s. This runs first every control step, so it also
    feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
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
    row["source"] = getattr(parts.get(PARTS[0]), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["turn_deg"]["bottle_a"], d["turn_deg"]["bottle_b"],
                      d["gap_mm"]["bottle_a"], d["gap_mm"]["bottle_b"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a bottle fell off the table or tipped over, or a bottle stands in the pan with its cap tight (backed off
    under 90 degrees); otherwise 1.0 both upright in the pan, both caps 90 to 360 degrees from tight on their threads,
    grippers clear and every part still; 0.75 the same but a cap over 360 degrees, or a gripper near, or a part moving;
    0.5 a cap off its thread; 0.25 one bottle upright in the pan with its cap backed off, the other not in the pan;
    0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score
