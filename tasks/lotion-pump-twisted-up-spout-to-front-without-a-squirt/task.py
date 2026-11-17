"""On the station table, real size: a 12 fl oz (355 ml) round white HDPE lotion bottle (66 mm across, 148 mm tall with
its black lock-down pump head, 0.42 kg) stands on a 3 mm rubber shelf liner at x 280 mm, y -250 mm, in front of a tiled
splash wall (x 430 to 450 mm). The pump head is locked down on the closure with its spout pointing -y. A quarter turn
anticlockwise (seen from above) frees the bayonet lugs and the pump spring lifts the head 23 mm; once it is free it
swivels either way, and pushed more than 4 mm down it makes a stroke (lotion squirted out).

Goal: the head turned past the lugs and risen, the spout swung round to face the robot (-x), the bottle standing where
it was, the head never pushed down once free, hands off. The turn takes about 0.035 N m, so the head has to be gripped
and turned; it then rises under its spring, so the hand has to follow it up while it turns, not press down.

The bottle (a loose body with two linked joints: a spring slider "lift" for the pump head's travel and, riding on it,
a hinge "twist" for its turn), the liner and the wall are mechkit compounds (mechkit.py, next to this file). Two declared
software rules (``_bayonet_rule``, a mechkit on_step callback): until the head has been turned 80 degrees the slider's
spring target holds the head down at the lock and the hinge's torsion spring pulls it back to 0; from then the lugs are
clear: the slider's target is the free height (the spring lifts the head) and the hinge holds where it is (a free
swivel). The scorer reads poses and joint positions. The arms are read through rigstate.py (the let-go check).

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
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import PART  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("The pump on the new lotion bottle is locked down. Twist the head to unlock it so it springs up, then "
               "turn the spout to point at you (-x). Do not press the head down once it is up and do not tip the bottle.")


# -- the parts --------------------------------------------------------------------------------------------------------
def bottle_shapes():
    c = mk.cylinder
    return [
        c((0.0, 0.0, L.BODY_H / 2), L.BODY_R, L.BODY_H, color=L.BOTTLE_WHITE),
        c((0.0, 0.0, L.BODY_H + L.SHOULDER_H / 2), L.SHOULDER_R, L.SHOULDER_H, color=L.BOTTLE_WHITE),
        c((0.0, 0.0, L.BODY_H + L.SHOULDER_H + L.COLLAR_H / 2), L.COLLAR_R, L.COLLAR_H, color=L.BOTTLE_WHITE),
    ]


def stem_shapes():
    z0 = L.BODY_H + L.SHOULDER_H  # the stem runs from inside the closure up to the free head's underside
    h = L.HEAD_UP - z0
    return [mk.cylinder((0.0, 0.0, z0 + h / 2), L.STEM_R, h, color=L.STEM_GREY)]


def head_shapes():
    top = L.HEAD_UP + L.HEAD_H
    mid = (L.SPOUT_R0 + L.SPOUT_R1) / 2
    return [
        mk.cylinder((0.0, 0.0, L.HEAD_UP + L.HEAD_H / 2), L.HEAD_R, L.HEAD_H, color=L.HEAD_BLACK),
        mk.box((0.0, -mid, top - L.SPOUT_T / 2), (L.SPOUT_W, L.SPOUT_R1 - L.SPOUT_R0, L.SPOUT_T), L.HEAD_BLACK),
        mk.box((0.0, -(L.SPOUT_R1 - L.NOZZLE / 2), top - L.SPOUT_T - L.NOZZLE_H / 2),
               (L.NOZZLE, L.NOZZLE, L.NOZZLE_H), L.HEAD_BLACK),
    ]


ANCHOR = (0.0, 0.0, L.HEAD_UP)
LIFT = mk.slider("lift", "-Z", stem_shapes(), upper=L.TRAVEL, lower=0.0, anchor=ANCHOR, mass=L.STEM_MASS,
                 spring=L.K_LIFT, damping=L.D_LIFT)
TWIST = mk.hinge("twist", "Z", ANCHOR, head_shapes(), upper=L.TWIST_LIMIT, lower=-L.TWIST_LIMIT, mass=L.HEAD_MASS,
                 spring=L.K_TWIST, damping=L.D_TWIST, parent="lift")

BOTTLE = mk.assembly(PART, origin=(L.BX, L.BY, L.TOP + L.BASE_Z), mass=L.BASE_MASS, shapes=bottle_shapes(),
                     joints=[LIFT, TWIST], color=L.BOTTLE_WHITE, friction=L.BOTTLE_FRICTION, tight_fit=True)
LINER = mk.assembly("liner", origin=(L.BX, L.BY, L.TOP),
                    shapes=[mk.box((0.0, 0.0, L.LINER_H / 2), (L.LINER_X, L.LINER_Y, L.LINER_H), L.LINER_GREY)],
                    color=L.LINER_GREY, friction=L.LINER_FRICTION, tight_fit=True)
SPLASH_WALL = mk.assembly(
    "splash_wall", origin=((L.WALL_X0 + L.WALL_X1) / 2, (L.WALL_Y0 + L.WALL_Y1) / 2, L.TOP),
    shapes=[mk.box((0.0, 0.0, L.WALL_H / 2), (L.WALL_X1 - L.WALL_X0, L.WALL_Y1 - L.WALL_Y0, L.WALL_H), L.WALL_TILE)],
    color=L.WALL_TILE, friction=0.4)
OBJECTS = [LINER, SPLASH_WALL, BOTTLE]
SPECS = {PART: BOTTLE}


# -- the declared rule: the bayonet and the pump spring ---------------------------------------------------------------
_RULE = {"free": False}


def bayonet_targets(twist: float, free: bool) -> tuple[bool, float, float]:
    """(free, lift target, twist target) for the head's turn ``twist`` (rad). The lugs clear at LUG_RAD and stay clear
    (the head is up; it cannot re-lock without being pressed down). Locked: the lift is held at the lock and the
    torsion spring pulls the head back to 0. Free: the spring lifts the head to its free height and the head holds
    where it is (swivel friction is the joint's damping)."""
    if not math.isfinite(twist):
        return free, L.TRAVEL if not free else 0.0, 0.0
    free = free or twist >= L.LUG_RAD
    if free:
        return True, 0.0, twist
    return False, L.TRAVEL, 0.0


@mk.on_step
def _bayonet_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    s = states.get(PART)
    try:
        twist = float(s.joints["twist"])
    except (AttributeError, KeyError, TypeError, ValueError):
        twist = float("nan")
    _RULE["free"], lift_t, twist_t = bayonet_targets(twist, _RULE["free"])
    mk.set_targets(BOTTLE, "lift", lift_t)
    mk.set_targets(BOTTLE, "twist", twist_t)


@mk.on_reset
def _rule_reset() -> None:
    _RULE["free"] = False


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped object as a box; it knows nothing of the bayonet or the pump. Offline only
# (never in the simulator), this model gives the scorer: a gripper closed on the head (its TCP within 12 mm of the
# axis, the pad tips alongside the head) turns it with the gripper's heading (read from the demo's own plan); the
# lugs clear at 80 degrees; once free, the head rides up with the gripper (to its free height) and, let go, springs to
# the free height; a closed block over a free head pushes it down; a locked head let go springs back to 0.
GRIP_LO, GRIP_HI = 0.10, 0.45  # the opening read on a 32 mm head (24 mm commanded: 0.26)
BLOCK_OPENING = 0.10
GRIP_XY = 0.012


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.twist = 0.0
        self.lift = L.TRAVEL
        self.free = False
        self.hold: dict = {}  # arm -> [tip height over the head's underside at the grip, last heading]
        self.k = 0
        self.plans = None

    def _plan_headings(self, t: float) -> dict:
        """Each planned arm's gripper heading (rad) at ``t`` from the demo's own waypoints (offline stub only)."""
        if self.plans is None:
            self.plans = {}
            try:
                from yam_arena import demo as stub_demo, station

                for arm, wps in dict(demo()).items():
                    self.plans[arm] = stub_demo.ArmPlan(list(wps), station.BASES[arm])
            except Exception:  # noqa: BLE001 - without the plans the head simply never turns
                self.plans = {}
        out = {}
        for arm, plan in self.plans.items():
            rot = plan.target(t)[1]
            out[arm] = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        return out

    def _bottom(self) -> float:
        """The head's underside over the table top."""
        return L.BASE_Z + L.HEAD_UP - self.lift

    def step(self, arms) -> None:
        self.k += 1
        heads = self._plan_headings(self.k / L.CONTROL_HZ)
        read = rigstate.read_arms(arms)
        if read is None:
            return
        gripped = False
        pressed = None
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            tip_h = float(tips[2]) - L.TOP
            near = math.hypot(float(tcp[0] - L.BX), float(tcp[1] - L.BY)) <= GRIP_XY
            bottom, top = self._bottom(), self._bottom() + L.HEAD_H
            heading = heads.get(side, 0.0)
            held = self.hold.get(side)
            if held is not None and not (GRIP_LO < opening < GRIP_HI and near):
                del self.hold[side]
                held = None
            if held is None and GRIP_LO < opening < GRIP_HI and near and bottom - 0.004 <= tip_h <= top - 0.004:
                held = self.hold[side] = [tip_h - bottom, heading]
            if held is not None:
                gripped = True
                self.twist += _wrap(heading - held[1])
                held[1] = heading
                if not self.free:
                    self.twist = max(0.0, self.twist)
                    if self.twist >= L.LUG_RAD:
                        self.free = True
                if self.free:
                    want = L.BASE_Z + L.HEAD_UP - (tip_h - held[0])
                    self.lift = float(np.clip(want, 0.0, L.TRAVEL))
            elif opening < BLOCK_OPENING and math.hypot(float(tips[0] - L.BX), float(tips[1] - L.BY)) <= GRIP_XY \
                    and tip_h < top:
                pressed = tip_h
        if not gripped:
            if pressed is not None and self.free:
                self.lift = float(np.clip(L.BASE_Z + L.HEAD_UP + L.HEAD_H - pressed, 0.0, L.TRAVEL))
            elif self.free:
                self.lift = 0.0
            else:
                self.twist, self.lift = 0.0, L.TRAVEL

    def states(self) -> dict:
        q = (0.0, 0.0, 0.0, 1.0)
        return {PART: mk.state_from_poses(BOTTLE, (L.BX, L.BY, L.TOP + L.BASE_Z, *q),
                                          joints={"lift": self.lift, "twist": self.twist}, source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(PART), "source", "sim") != "sim"


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
    """With TASK_TRACE=<file>, append one JSON line per control step: the bottle's pose, the head's turn and travel and
    what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The head turned past the lugs and 18 mm or more up, the spout within 15 degrees of -x, the bottle standing on
    its mark, never a stroke or a tip, both grippers 30 mm clear and the bottle still for 1 s; held 1 s. This runs
    first every control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {PART: objects.get(PART)}
    arms = rigstate.arms()
    if no_simulator(parts):
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, "rule_free": _RULE["free"], **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(PART), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the bottle tipped past 45 degrees or fell off, or the head was pushed down more than 4 mm once free, or
    unless the grippers are 30 mm clear and the bottle still; then, with the bottle standing: 1.0 head turned past
    the lugs and 18 mm or more up with the spout within 15 degrees of -x; 0.5 the same, spout elsewhere; 0.25 not
    unlocked but the head turned 30 degrees at some moment; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
# The wrist cannot turn the head through its whole 270 degree path in one grip (joint 6 allows about -100 to +120
# degrees of heading), so the head is turned in three grips: unlock, then two swings, the open hand lifted clear of the
# spout and turned in the air between them. The pads always close across the spout (heading = the spout's bearing).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
GRIP_F = -(2 * L.HEAD_R - 0.008) / 2  # 24 mm commanded across the 32 mm head
PSI0 = -math.pi / 2  # pads close along x, clear of the spout (-y)
UNLOCK_TURN = math.radians(95.0)  # past the 80 degree lugs: the spout ends at +5 degrees
SWING_1 = -math.radians(90.0)
SWING_2 = -math.radians(95.0)  # total head turn -90 degrees: the spout from -y round to -x
HOVER = 0.200  # TCP: pad tips 12 mm over the free head's top (the spout)
GRIP_TIPS = L.BASE_Z + L.HEAD_BOTTOM_LOCKED + 0.009  # pad tips 9 mm over the locked head's underside
GRIP_TCP = GRIP_TIPS + PAD_TIP
RISE = 0.022  # the grip follows the head up this far (commanded; the arm trails)
FREE_TCP = L.BASE_Z + L.HEAD_UP + 0.009 + PAD_TIP  # the same grip on the free head
PARK = (0.20, -0.44, 0.15)


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _grip(plan, z, psi, turn, label, turn_s, rise=0.0, press=0.0):
    """Down round the head at heading ``psi``, close, turn it ``turn`` (rad), optionally follow it up ``rise``, open,
    up to the hover."""
    x, y = L.BX, L.BY
    end = psi + turn
    plan += [
        _w(2.0, x, y, z, OPEN, f"down round the head ({label})", psi),
        _w(0.5, x, y, z, OPEN, "settle", psi),
        _w(1.5, x, y, z, GRIP_F, "grip the head across its 32 mm", psi),
        _w(turn_s, x, y, z, GRIP_F, f"turn it ({label})", end),
    ]
    if rise:
        plan += [_w(1.5, x, y, z + rise, GRIP_F, "follow the head up as it springs", end)]
    if press:
        plan += [_w(1.5, x, y, z + rise - press, GRIP_F, "press the head down (never in the demo)", end)]
    plan += [
        _w(0.5, x, y, plan[-1].pos[2] - L.TOP, GRIP_F, "settle", end),
        _w(1.5, x, y, plan[-1].pos[2] - L.TOP, OPEN, "let go", end),
        _w(1.5, x, y, HOVER, OPEN, "straight up", end),
    ]
    return end


def plan_right(turn: float = UNLOCK_TURN, swings: bool = True, press: float = 0.0):
    """The right arm: grip 1 turns the head past the lugs and follows it up as it springs; grips 2 and 3 swing the
    spout round to -x. Ends parked aside."""
    x, y = L.BX, L.BY
    plan = [_w(2.5, x, y, HOVER, OPEN, "over the pump head", PSI0)]
    psi = _grip(plan, GRIP_TCP, PSI0, turn, "unlock", 3.0, rise=RISE if turn >= L.LUG_RAD else 0.0, press=press)
    if turn >= L.LUG_RAD and swings and not press:
        psi = _grip(plan, FREE_TCP, psi, SWING_1, "swing", 2.5)
        plan += [_w(2.0, x, y, HOVER, OPEN, "turn the open hand to the spout's other side", psi + math.pi)]
        psi = _grip(plan, FREE_TCP, psi + math.pi, SWING_2, "swing", 2.5)
    plan += [_w(1.5, *PARK, OPEN, "aside", psi)]
    return plan


# Wrong ways (counterfactuals): name -> plan arguments.
VARIANT_ARGS = {
    "half_turn": {"turn": math.radians(45.0)},  # turned 45 degrees and let go: it springs back (0.25)
    "no_swing": {"swings": False},  # unlocked, spout left at +x (0.5)
    "press": {"press": 0.030},  # pushed the head down once free (0.0)
}
VARIANTS = tuple(VARIANT_ARGS)


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    return {"right": plan_right(**VARIANT_ARGS.get(variant, {}))}


def demo():
    """The right arm alone does the task (the left stays at its ready pose). DEMO_HOLD_ONLY=1: no plan (both arms hold
    the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      half_turn  the head turned 45 degrees and let go: it springs back locked (0.25)
      no_swing   unlocked, the spout left pointing +x (0.5)
      press      the head pushed down once free (0.0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
