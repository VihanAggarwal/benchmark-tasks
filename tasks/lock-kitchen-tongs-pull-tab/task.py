"""On the station worktop, real size: a pair of OXO Good Grips 9-inch locking tongs (230 mm, 135 g) lying open on its
side, hinge pin upright, nothing else on the worktop and nothing fixed. Three bodies on one loose articulation: the
arm that carries the pin and spring (the root, 61 g), the other arm on the hinge (54 g; a joint spring holds it
27.5 deg open: 0.20 N m there, 0.40 N m shut) and the black pull tab on a slider along the pin arm (20 g, 12 mm of
travel, 1 N of friction). The tab carries a steel lock block; drawn out with the arms within 2.86 deg of shut, the
block slides under two lugs on the other arm's heel and, let go, the arms spring back onto it and stay shut. With the
arms open the lugs stand in the block's way and stop the tab 3.8 to 6.3 mm out. The lock is geometry: contact between
the block (on the tab) and the lugs (on the free arm), no code rule.

Goal: squeeze the arms shut, draw the tab out to its stop, take both grippers away; the lock then holds the tongs shut
on the worktop. See scoring.py for the tiers.

The tongs are a mechkit assembly (mechkit.py next to this file); the arms are read through rigstate.py. Neither
changes what the embodiment does. One override lives here (see ``_usd_plan_self_collisions``): this articulation is
written with PhysX self-collision on, so the tab's block and the free arm's lugs (two sibling links) touch.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assembly is built as an articulation)
import rigstate  # noqa: E402
from offline_kinematics import OfflineTongs  # noqa: E402
from scoring import HINGE, TAB, TONGS, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Lock the kitchen tongs shut. Squeeze the two steel arms together, pull the black tab at the hinge "
               "end straight out until it stops, then take both grippers away. The lock must then hold the tongs "
               "shut on the worktop.")


# -- self-collision for this articulation ----------------------------------------------------------------------------
# mechkit writes every articulation with PhysX self-collision off (``usd_plan(...)["self_collisions"]`` is False)
# and its author step already writes ``enabledSelfCollisions`` from that plan. The lock needs the tab's block (one
# link) to touch the free arm's lugs (a sibling link), so for THIS assembly only the plan says True. PhysX still
# filters every parent-child pair (pin arm and free arm, pin arm and tab), so the block against the lugs (and the
# rest of the tab against the rest of the free arm, 1.4 mm or more apart everywhere else) is the only new pair.
# mechkit.py itself is a byte-for-byte copy of tools/kit/mechkit.py.
_KIT_USD_PLAN = getattr(mk.usd_plan, "_kit_original", mk.usd_plan)


def _usd_plan_self_collisions(spec, root):
    plan = _KIT_USD_PLAN(spec, root)
    if getattr(spec, "name", None) == TONGS:
        plan["self_collisions"] = True
    return plan


_usd_plan_self_collisions._kit_original = _KIT_USD_PLAN
mk.usd_plan = _usd_plan_self_collisions  # mechkit's author step reads usd_plan from its own module at spawn time


# -- the tongs ------------------------------------------------------------------------------------------------------
def _box(dims, color, turn: float = 0.0):
    """A box from tongs-frame ranges (u0, u1, v0, v1, w0, w1 in mm) in the kit's base frame, turned ``turn`` rad about
    the pin axis (the base z axis)."""
    u0, u1, v0, v1, w0, w1 = dims
    x, y, z = ((w0 + w1) / 2 - L.PIN_W) * L.MM, (u0 + u1) / 2 * L.MM, ((v0 + v1) / 2 + 15.0) * L.MM
    size = ((w1 - w0) * L.MM, (u1 - u0) * L.MM, (v1 - v0) * L.MM)
    if not turn:
        return mk.box((x, y, z), size, color)
    c, s = math.cos(turn), math.sin(turn)
    return mk.box((c * x - s * y, s * x + c * y, z), size, color, ypr=(turn, 0.0, 0.0))


_COLOURS = {"bar": L.STEEL, "head": L.STEEL, "ear_up": L.DARK_STEEL, "ear_down": L.DARK_STEEL,
            "lug_up": L.DARK_STEEL, "lug_down": L.DARK_STEEL, "block": L.DARK_STEEL, "stem": L.BLACK}
PIN_ARM = [_box(d, _COLOURS[k]) for k, d in L.PIN_ARM_BOXES.items()]
FREE_ARM = mk.hinge(HINGE, "Z", anchor=(0.0, 0.0, 0.015), mass=L.FREE_ARM_MASS, upper=L.OPEN,
                    spring=L.SPRING_RATE, damping=L.HINGE_DAMPING,
                    shapes=[_box(d, _COLOURS[k], -L.OPEN) for k, d in L.FREE_ARM_BOXES.items()])
TAB_LINK = mk.slider(TAB, "-Y", mass=L.TAB_MASS, upper=L.TAB_TRAVEL, damping=L.TAB_DAMPING, friction=L.TAB_FRICTION,
                     shapes=[_box(d, _COLOURS.get(k, L.BLACK)) for k, d in L.TAB_BOXES.items()])
_HEAD = L.PIN_ARM_BOXES["head"]
# Offline the tongs are their pin arm's head block (nothing grips it), and offline_kinematics.py moves the rest.
_HEAD_CENTRE = np.array([((_HEAD[4] + _HEAD[5]) / 2 - L.PIN_W) * L.MM, (_HEAD[0] + _HEAD[1]) / 2 * L.MM,
                         ((_HEAD[2] + _HEAD[3]) / 2 + 15.0) * L.MM])
_HEAD_SIZE = ((_HEAD[5] - _HEAD[4]) * L.MM, (_HEAD[1] - _HEAD[0]) * L.MM, (_HEAD[3] - _HEAD[2]) * L.MM)


class Tongs(mk.Assembly):
    """The tongs exactly as mechkit builds and reads them in the simulator. Only tools/offline/check_task.py calls the
    two methods below: offline the tongs are the pin arm's head block, which no gripper touches, so the checker's own
    carry proxy leaves them alone; ``offline_state`` puts the tongs' frame back on that block."""

    def offline_box(self):
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ _HEAD_CENTRE
        return tuple(float(v) for v in centre), tuple(float(v) for v in _HEAD_SIZE), float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ _HEAD_CENTRE
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


TONGS_SPEC = Tongs(name=TONGS, pos=L.ORIGIN, color=L.STEEL, mass=L.PIN_ARM_MASS, yaw=L.YAW, shapes=tuple(PIN_ARM),
                   joints=(FREE_ARM, TAB_LINK), friction=L.FRICTION, **mk.TIGHT_FIT)
TONGS_SPEC._check()
OBJECTS = [TONGS_SPEC]


# -- the hinge spring's preload --------------------------------------------------------------------------------------
# The spring pulls toward a target 0.480 rad beyond the open stop: 0.20 N m at the stop, 0.40 N m shut. mechkit sets
# every target back to 0 at reset, so it is set again after each reset and at every step (a constant, not a rule).
@mk.on_step
def _spring_preload(states) -> None:
    mk.set_targets(TONGS_SPEC, HINGE, L.SPRING_TARGET)


@mk.on_reset
def _spring_preload_reset() -> None:
    mk.set_targets(TONGS_SPEC, HINGE, L.SPRING_TARGET)
    OFFLINE.reset()


_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineTongs()  # only stepped without a simulator (see offline_kinematics.py)


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the tongs: mechkit has no environment AND the state handed in was not read
    from one (mechkit marks those ``source="sim"``). Either sign of a simulator keeps its states untouched."""
    return mk.env() is None and getattr(objects.get(TONGS), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    """Without a simulator nothing moves the tongs' joints, so offline_kinematics.py moves them from the arms' motion;
    the scorer then sees those states."""
    OFFLINE.step(arms)
    origin, yaw = OFFLINE.pose()
    out = dict(objects)
    out[TONGS] = mk.state_from_poses(TONGS_SPEC, (*origin, *mk.yaw_quat(yaw)), np.zeros(6), joints=OFFLINE.joints(),
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
    """Locked, let go, still and on the worktop for the scorer's 1 s windows. This runs every control step, so it
    also feeds the tracker (and, without a simulator, the offline kinematics that move the tongs)."""
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
        row["offline"] = {"hold": OFFLINE.hold, "ring": OFFLINE.ring, "towed_mm": round(OFFLINE.towed * 1000, 1)}
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 locked (tab within 1 mm of its stop, tips within 14 mm), let go, still and on the worktop; 0.5 locked but
    not all of the rest; 0.25 squeezed shut for 1 s at some point but not locked at the end; 0 otherwise. The score
    of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. Pad-tip points; the TCP is
# 13 mm above. Both grippers point straight down with their fingers closing along y (yaw 0), across the tongs, which
# lie along x with their tips toward the robot. The right gripper holds the arms shut 90 mm from the pin (near its own
# base line), the left one draws the tab 14 cm further out. Order matters for the arms' own clearance: the right
# gripper goes down first and rises last; the left one comes and goes from above while the right one is down (their
# collision capsules: 25 mm apart or more while both are down, more the higher the left one is).
# Finger targets: the joint value is minus half the commanded gap.
PAD_TIPS = 0.013
YAW = 0.0
FULL_OPEN = -0.04695  # 94 mm
HOLD_CLOSE = -0.012  # 24 mm commanded on arms that meet at 28 mm: they squeeze shut against the 4.4 N needed there
RELEASE_OPEN = -0.030  # 60 mm
RING_OPEN = -0.020  # 40 mm: 16 mm a side of the 8 mm ring
RING_CLOSE = -0.0025  # 5 mm commanded on the 8 mm ring
HOVER = 0.060  # pad tips over the worktop on the way in
HOLD_TIPS = 0.006  # the right pad tips beside the arms (5 to 25 mm up)
RING_TIPS = 0.004  # the left pad tips beside the ring (3 to 27 mm up)
LIFT_RING, LIFT_HOLD = 0.120, 0.080
# The pull is commanded well past the 12 mm stop. Both arms are soft along x at these poses (joints 4 to 6 at
# 10 N m/rad, 1 to 3 at 80): at the pad tips about 6.4 mm per newton for the left arm and 5.0 mm per newton for the
# right one (yamkin Jacobian, tested), so the tab's 1 N of friction alone takes up about 11 mm of commanded travel
# (the left pads lag, the right pads and the tongs give way along +x). A pull commanded to 12 or 13.5 mm would leave
# the tab short of the 11 mm that locks. Past the stop the extra travel only presses the tab on its stop with about
# 1 to 3 N, far below what either pinch holds (about 9 N on the ring, 12 N on the arms).
PULL_SLACK = 0.015
PULL = L.TAB_TRAVEL + PULL_SLACK  # m along +x, commanded: the pads from x 0.240 to 0.267
PULL_S, PULL_DWELL_S = 3.0, 1.0  # 9 mm/s commanded, then a dwell for the arms to settle on the stop
HOME = {"left": (0.30, 0.22, 0.12), "right": (0.30, -0.22, 0.12)}  # pad-tip points, clear of everything
PARK_LEFT = (0.26, 0.14, 0.12)  # the left gripper waits here while the right one goes down


def _at(x, y, tips_h):
    return (x, y, L.TOP + tips_h + PAD_TIPS)


def _hold_plan(release: bool = True, release_early: bool = False) -> list[Waypoint]:
    """Right gripper: hold the arms shut 90 mm from the pin until the tab is out and the left gripper has risen,
    open, rise and go home (26.5 s)."""
    x, y = L.HOLD_XY
    plan = [
        Waypoint(2.5, _at(x, y, HOVER), FULL_OPEN, "over the arms 90 mm from the pin, open 94 mm", YAW),
        Waypoint(1.5, _at(x, y, HOLD_TIPS), FULL_OPEN, "slowly down, one pad outside each open arm", YAW),
        Waypoint(0.5, _at(x, y, HOLD_TIPS), FULL_OPEN, "settle", YAW),
        Waypoint(2.0, _at(x, y, HOLD_TIPS), HOLD_CLOSE, "squeeze the arms shut (commanded 24 mm)", YAW),
        Waypoint(1.0, _at(x, y, HOLD_TIPS), HOLD_CLOSE, "hold: the arms settle shut along x", YAW),
    ]
    if release_early:  # counterfactual: let go before the tab is drawn
        return plan + [
            Waypoint(1.5, _at(x, y, HOLD_TIPS), RELEASE_OPEN, "open: the arms spring back", YAW),
            Waypoint(1.5, _at(x, y, LIFT_HOLD), RELEASE_OPEN, "straight up", YAW),
            Waypoint(3.0, _at(*HOME["right"]), RELEASE_OPEN, "home", YAW),
        ]
    plan.append(Waypoint(13.0, _at(x, y, HOLD_TIPS), HOLD_CLOSE, "keep holding until the left gripper has risen", YAW))
    if not release:  # counterfactual: never let go
        return plan + [Waypoint(6.0, _at(x, y, HOLD_TIPS), HOLD_CLOSE, "keep holding to the end", YAW)]
    return plan + [
        Waypoint(1.5, _at(x, y, HOLD_TIPS), RELEASE_OPEN, "open to 60 mm: the arms spring onto the lock", YAW),
        Waypoint(1.5, _at(x, y, LIFT_HOLD), RELEASE_OPEN, "straight up, clear of the tongs", YAW),
        Waypoint(3.0, _at(*HOME["right"]), RELEASE_OPEN, "home", YAW),
    ]


def _ring_plan(pull: float = PULL, park: float = 2.0, wait: float = 3.0, ring_xy=L.RING_GRIP_XY) -> list[Waypoint]:
    """Left gripper: wait clear, pinch the ring, draw the tab out to its stop, open, rise and go home (23.5 s)."""
    x, y = ring_xy
    return [
        Waypoint(2.5, _at(*PARK_LEFT), RING_OPEN, "wait clear of the right arm, open 40 mm", YAW),
        Waypoint(park, _at(*PARK_LEFT), RING_OPEN, "wait while the right gripper goes down", YAW),
        Waypoint(2.0, _at(x, y, HOVER), RING_OPEN, "over the ring", YAW),
        Waypoint(wait, _at(x, y, HOVER), RING_OPEN, "wait for the arms to be held shut", YAW),
        Waypoint(1.5, _at(x, y, RING_TIPS), RING_OPEN, "slowly down, one pad either side of the ring", YAW),
        Waypoint(0.5, _at(x, y, RING_TIPS), RING_OPEN, "settle", YAW),
        Waypoint(1.5, _at(x, y, RING_TIPS), RING_CLOSE, "pinch the ring (commanded 5 mm)", YAW),
        Waypoint(0.5, _at(x, y, RING_TIPS), RING_CLOSE, "settle the pinch", YAW),
        Waypoint(PULL_S, _at(x + pull, y, RING_TIPS), RING_CLOSE, "draw the tab straight out to its stop", YAW),
        Waypoint(PULL_DWELL_S, _at(x + pull, y, RING_TIPS), RING_CLOSE, "dwell at the stop", YAW),
        Waypoint(1.5, _at(x + pull, y, RING_TIPS), RING_OPEN, "let go of the ring", YAW),
        Waypoint(1.5, _at(x + pull, y, LIFT_RING), RING_OPEN, "straight up, clear of the tongs", YAW),
        Waypoint(3.0, _at(*HOME["left"]), RING_OPEN, "home", YAW),
    ]


VARIANTS = ("tab_only", "squeeze_release_pull", "tab_8mm", "no_let_go")


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm checks).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      tab_only               the left arm alone pinches the ring where it lies at the start and makes the full
                             commanded pull (27 mm along +x)
      squeeze_release_pull   the right squeezes for 1 s and lets go, then the left pulls
      tab_8mm                the full demo with the pull commanded to 8 mm only (the trace's tab_mm says where the
                             tab stopped; the arms' give takes up part of it)
      no_let_go              the full demo, but the right gripper never opens"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "tab_only":
        start = L.to_world([(L.RING_GRIP_U, L.RING_CENTRE_UW[1])])[0]  # on the open tongs: (0.241, 0.006)
        plan = {"left": _ring_plan(park=0.5, wait=0.5, ring_xy=(float(start[0]), float(start[1])))}
    elif variant == "squeeze_release_pull":
        plan = {"right": _hold_plan(release_early=True), "left": _ring_plan(park=9.0, wait=0.5)}
    elif variant == "tab_8mm":
        plan = {"right": _hold_plan(), "left": _ring_plan(pull=0.008)}
    elif variant == "no_let_go":
        plan = {"right": _hold_plan(release=False), "left": _ring_plan()}
    else:
        plan = {"right": _hold_plan(), "left": _ring_plan()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
