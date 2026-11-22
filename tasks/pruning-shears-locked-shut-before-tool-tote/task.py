"""On the station bench, real size: one loose pair of Fiskars SingleStep P26 bypass pruning shears (model
111260: 20.8 cm long, 7.3 cm closed width, 2.2 cm thick, 200 g) lying flat and open at 20.5 deg (22 mm
blade tip opening), black-and-orange handles toward the robot, bright curved bypass blade exposed forward;
and an open-top green polypropylene garden tool tote (26 x 18 x 6 cm interior) fixed on the robot's right
at (0.280, -0.200).

Single loose articulation (fixed_base=False) for the pruning shears (layout.py):
- Base body (105 g): lower handle, lower orange bolster, pivot housing, catch rail, anvil counter-blade.
- Hinge ``jaw`` (85 g, about +Z at pivot): upper handle, upper orange bolster, pivot plate, curved bypass
  blade; 0 rad = 20.5 deg open, +0.3578 rad = 0.0 deg closed; return spring 1.40 N m/rad (4.5 N at grip).
- Slider ``catch`` (10 g, along -Y on top of pivot housing): orange top-mounted linear sliding thumb safety
  catch, 12 mm travel, 166.7 N/m detent spring (2.0 N est).

Declared interlock rule (``_interlock_step``, a mechkit @mk.on_step callback using interlock.py):
when handles are closed (open_deg <= 5 deg) AND thumb safety catch is slid >= 9 mm, the catch latches
at 12 mm and the handle hinge drive target switches to closed (0 deg) so the shears stay locked shut
after release. Sliding the catch while handles are open (> 5 deg) does not engage.

Goal: squeeze the handles closed, engage the thumb safety catch, place the locked shears flat inside the
garden tool tote, and retract both grippers >= 4 cm clear for 1.0 s. Putting unlocked shears below the
6.0 cm tote rim or touching the exposed blade edge while open latches a hard 0.

Run:     inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) leaves that arm at its ready pose.
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
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
from interlock import Interlock  # noqa: E402
from offline_shears import OfflineShears  # noqa: E402
from scoring import CATCH, JAW, SHEARS, TOTE, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Put the bypass pruning shears away in the open garden tool tote."


# -- assemblies ----------------------------------------------------------------------------------------------------
def _mk_box(centre, size, yaw_deg, color):
    if abs(yaw_deg) < 1e-9:
        return mk.box(centre, size, color)
    return mk.box(centre, size, color, ypr=(math.radians(yaw_deg), 0.0, 0.0))


BASE_SHAPES = (
    [_mk_box(c, s, y, col) for _, c, s, y, col in L.base_box_specs()]
    + [mk.cylinder((0.0, 0.0, L.PIVOT_H / 2), 0.005, L.PIVOT_H, color=L.STEEL_DARK)]
)
UPPER_SHAPES = [_mk_box(c, s, y, col) for _, c, s, y, col in L.upper_box_specs(L.THETA_OPEN_DEG)]
# mechkit draws every link shape in the BASE frame (the anchor only sets the link frame's origin). GPU runs 1-5 drew
# the tab at (0, 0, 0.005): inside the pivot boss at bench level, invisible in every frame, so the left pusher swept
# through air at z 25 mm and the catch never moved (job 20261122T041243: catch_mm <= 0.4 while the handles were shut).
CATCH_SHAPES = [mk.box((L.CATCH_ANCHOR[0], L.CATCH_ANCHOR[1], L.CATCH_ANCHOR[2] + L.CATCH_SIZE[2] / 2),
                       L.CATCH_SIZE, L.ACCENT_ORANGE)]

JAW_JOINT = mk.hinge(
    JAW, "Z", (0.0, 0.0, 0.0), UPPER_SHAPES,
    lower=L.JAW_LOWER, upper=L.JAW_UPPER, mass=L.UPPER_MASS,
    spring=L.JAW_SPRING, damping=L.JAW_DAMPING, friction=L.JAW_FRICTION,
)
CATCH_JOINT = mk.slider(
    CATCH, "-Y", CATCH_SHAPES, anchor=L.CATCH_ANCHOR,
    lower=0.0, upper=L.CATCH_TRAVEL, mass=L.CATCH_MASS,
    spring=L.CATCH_SPRING, damping=L.CATCH_DAMPING, friction=L.CATCH_FRICTION,
)

# Offline proxy box at the anvil (no gripper grabs it; offline_shears.py moves the shears)
_ANVIL_CENTRE = np.array([0.040, 0.003, 0.005])
_ANVIL_SIZE = (0.040, 0.012, 0.010)


class Shears(mk.Assembly):
    def offline_box(self):
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ _ANVIL_CENTRE
        return tuple(float(v) for v in centre), tuple(float(v) for v in _ANVIL_SIZE), float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ _ANVIL_CENTRE
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


SHEARS_SPEC = Shears(
    name=SHEARS, pos=L.SHEARS_ORIGIN, color=L.HANDLE_BLACK, mass=L.BASE_MASS, yaw=L.SHEARS_YAW,
    shapes=tuple(BASE_SHAPES), joints=(JAW_JOINT, CATCH_JOINT), friction=L.FRICTION,
    solver_iterations=32,
)
SHEARS_SPEC._check()

TOTE_SHAPES = [mk.box(c, s, col) for _, c, s, col in L.tote_box_specs()]
TOTE_SPEC = mk.Assembly(
    name=TOTE, pos=(L.TOTE_CENTRE[0], L.TOTE_CENTRE[1], L.TOP), color=L.TOTE_GREEN,
    mass=None, shapes=tuple(TOTE_SHAPES), friction=0.6,
)
TOTE_SPEC._check()

OBJECTS = [SHEARS_SPEC, TOTE_SPEC]


# -- interlock rule (simulator @mk.on_step) ------------------------------------------------------------------------
INTERLOCK = Interlock()


@mk.on_step
def _interlock_step(states) -> None:
    s = states.get(SHEARS)
    try:
        q_jaw = float(s.joints[JAW])
        q_catch = float(s.joints[CATCH])
    except (AttributeError, KeyError, TypeError, ValueError):
        q_jaw = q_catch = math.nan
    locked, jaw_tgt, catch_tgt = INTERLOCK.step(q_jaw, q_catch)
    mk.set_targets(SHEARS_SPEC, JAW, jaw_tgt)
    mk.set_targets(SHEARS_SPEC, CATCH, catch_tgt)


@mk.on_reset
def _interlock_reset() -> None:
    INTERLOCK.reset()
    mk.set_targets(SHEARS_SPEC, JAW, 0.0)
    mk.set_targets(SHEARS_SPEC, CATCH, 0.0)
    OFFLINE.reset()


_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineShears()


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    INTERLOCK.reset()
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(SHEARS), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    OFFLINE.step(arms)
    origin, yaw = OFFLINE.pose()
    out = dict(objects)
    out[SHEARS] = mk.state_from_poses(
        SHEARS_SPEC, (*origin, *mk.yaw_quat(yaw)), np.zeros(6),
        joints=OFFLINE.joints(), source="offline",
    )
    return out


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            INTERLOCK.reset()
            OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    offline = no_simulator(objects)
    if offline:
        objects = offline_states(objects, arms)
        locked_flag = OFFLINE.interlock.locked
    else:
        locked_flag = INTERLOCK.locked
    result = _TRACKER.update(objects, arms, locked_flag)
    row = {
        "episode": rigstate.episode(),
        "step": _TRACKER.steps,
        "score": result.score,
        "success": result.success,
        "fallback_resets": _TRACKER.resets,
        **result.details,
        "interlock": {"locked": locked_flag, "switches": INTERLOCK.switches, "skipped": INTERLOCK.skipped},
    }
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
    return _TRACKER.last.score


# -- scripted demo -------------------------------------------------------------------------------------------------
# Right arm straddles both open handles from above at x_local -104 mm (x 0.156), squeezes them shut and holds them
# against the 4.5 N return spring while the left arm, coming in 4 cm ahead of the catch, slides the thumb safety catch
# 12 mm along -y to its stop (commanded 20 mm past it for arm lag) with its closed pads (TCP x 0.278). Left lifts, moves ahead and retracts; then
# right lifts the locked shears over the 6.0 cm rim, carries them to x 0.217 / y -0.200, lowers them flat onto the
# tote floor, opens and retracts.
F_STRADDLE = -0.04695  # 93.9 mm full opening (straddles <= 77.5 mm open handle span under the pads, ~8 mm per side)
F_RELEASE = -0.042  # 84 mm opening for clean release inside tote
F_SQUEEZE = -0.011  # 22 mm commanded on 38 mm closed handles (~8 mm squeeze per side for 5000 N/m finger clamp)
F_PUSH = 0.0  # closed pads as rigid pusher for thumb catch
YAW_R = 0.0  # yaw 0 closes the fingers along world Y (across the handles). GPU runs 1-3 used pi/2 = closing along X (along the handles): pads closed on nothing, handles never moved
YAW_L = 0.0  # fingers close along Y: closed pair 16 mm along Y (the push direction), 27 mm along X (covers the 12 mm tab). Runs 1-5 used -pi/2 (27 mm along Y)

# GPU run 4 (job 20261122T034047, yaw 0): at x_local -0.112 the 20 mm wide pads reach back to x_local -0.122,
# where the open handles span 80 mm, leaving ~5 mm on the +y side; the +y pad landed ON the upper handle (right TCP
# stuck 10.5 mm high, z 0.776 vs 0.7655, steps 64-99), dragged it to 13.8 deg, twisted the shears -6.9 deg, slid off
# and closed to 22 mm around the lower handle alone (step 106+). Grip 8 mm nearer the pivot and centre on the widest
# span under the pads (77.5 mm: ~8.2 mm per side at 94 mm). GPU run 5 confirmed it: open_deg 20.5 -> 0 by step 106,
# opening stopped at 0.409 (38 mm), TCP at z 0.766. Not further forward: the wrists (link4) meet when the TCPs are
# < ~12 cm apart in x while the left pushes the catch.
DEMO_GRIP_X_LOCAL = -0.104  # demo squeeze point (pad covers x_local -0.114..-0.094); see note above
PAD_HALF_W = 0.010  # finger pad half width along the shears (x)
X_GRIP = L.SHEARS_ORIGIN[0] + DEMO_GRIP_X_LOCAL
# Place: the shears run from 41 mm behind the grip (handle tails) to 167 mm ahead (blade tips), so put the grip
# 63 mm behind the tote centre: shears x tote_x -0.104..+0.104 inside the 26 cm interior (26 mm each end).
X_PLACE = L.TOTE_CENTRE[0] - 0.5 * ((L.LENGTH_FRONT - DEMO_GRIP_X_LOCAL) - (L.LENGTH_BACK + DEMO_GRIP_X_LOCAL))
_, _SPAN_REAR, _ = L.handle_outer_span_at(DEMO_GRIP_X_LOCAL - PAD_HALF_W, L.THETA_OPEN_DEG)  # widest under the pad
_LOWER_OUT = -(L.HANDLE_Y_OFF + L.HANDLE_W / 2.0)
_Y_MID_OPEN = _LOWER_OUT + _SPAN_REAR / 2.0  # centre the open pads on the widest span under them
Y_GRIP_OPEN = L.SHEARS_ORIGIN[1] + _Y_MID_OPEN  # ~+0.0347 m
Y_GRIP_SHUT = L.SHEARS_ORIGIN[1]  # +0.015 m

H_GRIP = 0.018  # pad tips 5 mm above bench, squarely beside the 16 mm handles
H_OVER = 0.105  # hover above handles
H_CARRY = 0.125  # carry height: 65 mm above 60 mm tote rim, well below 165 mm limit
H_PLACE = 0.026  # pad tips 13 mm above bench (9 mm above 4 mm tote floor; shears settle 4 mm on release)

X_CATCH = L.SHEARS_ORIGIN[0] + L.CATCH_ANCHOR[0] + 0.004  # 27 mm pads cover the 12 mm tab with 3.5 mm behind, 11.5 ahead; +4 mm for arm room
X_STAGE = X_CATCH + 0.036  # transit 4 cm ahead of the catch: the left gripper passes well in front of the right wrist
Y_CATCH_CLEAR = 0.180  # sideways clearance waypoint on the left's own side
Y_CATCH_START = L.SHEARS_ORIGIN[1] + 0.035  # -Y pad face (8 mm from TCP) 15 mm clear of the tab +Y face (origin y + 0.012)
# GPU run 6 (job 20261122T043513): the tab is now hit (catch_mm 0.6 -> 6.9 at steps 257-281) but stalls short of the
# 9 mm engage: the left TCP stopped at y 0.022 against a 0.016 command (6 mm lag under the 1.4 N detent load) and
# ~3.7 mm more is lost to pad/shears give (TCP moved 10 mm from first contact, the catch 6.3 mm). Linear fit of that
# run: 12 mm needs a command near y 0.004, so command 13 mm deeper than run 6.
Y_CATCH_END = L.SHEARS_ORIGIN[1] - 0.012  # -Y pad face 20 mm past the locked +Y face in the stub; ~0 to +3 mm on the GPU after lag
H_CATCH = 0.039  # pad tips at 26 mm: on the 22..32 mm tab, 4 mm above the 22 mm catch rail / pivot pin tops
H_CATCH_OVER = 0.060

HOME_L = (0.240, +0.250, L.TOP + 0.120)
HOME_R = (0.240, -0.250, L.TOP + 0.125)


def _at(x: float, y: float, h: float) -> tuple[float, float, float]:
    return (float(x), float(y), L.TOP + float(h))


def _right_demo(place_in_tote: bool = True) -> list[Waypoint]:
    plan = [
        Waypoint(1.5, _at(X_GRIP, Y_GRIP_OPEN, H_OVER), F_STRADDLE, "hover over open handles, 94 mm open", YAW_R),
        Waypoint(0.5, _at(X_GRIP, Y_GRIP_OPEN, H_OVER), F_STRADDLE, "settle full 94 mm opening before descent", YAW_R),
        Waypoint(1.2, _at(X_GRIP, Y_GRIP_OPEN, H_GRIP + 0.020), F_STRADDLE, "descend to 2 cm above straddle height", YAW_R),
        Waypoint(1.0, _at(X_GRIP, Y_GRIP_OPEN, H_GRIP), F_STRADDLE, "slow vertical final descent beside open handles", YAW_R),
        Waypoint(0.5, _at(X_GRIP, Y_GRIP_OPEN, H_GRIP), F_STRADDLE, "settle around open handles", YAW_R),
        Waypoint(1.8, _at(X_GRIP, Y_GRIP_SHUT, H_GRIP), F_SQUEEZE, "squeeze handles closed against spring", YAW_R),
        Waypoint(0.7, _at(X_GRIP, Y_GRIP_SHUT, H_GRIP), F_SQUEEZE, "settle closed grip", YAW_R),
        Waypoint(13.0, _at(X_GRIP, Y_GRIP_SHUT, H_GRIP), F_SQUEEZE, "hold handles shut while left engages catch", YAW_R),
    ]
    if not place_in_tote:
        plan += [
            Waypoint(1.5, _at(X_GRIP, Y_GRIP_SHUT, H_GRIP), F_STRADDLE, "release locked shears on bench", YAW_R),
            Waypoint(1.5, _at(X_GRIP, Y_GRIP_SHUT, H_OVER), F_STRADDLE, "lift clear", YAW_R),
            Waypoint(1.5, HOME_R, F_RELEASE, "retract home", 0.0),
        ]
        return plan
    plan += [
        Waypoint(2.0, _at(X_GRIP, Y_GRIP_SHUT, H_CARRY), F_SQUEEZE, "lift locked shears above tote rim", YAW_R),
        Waypoint(2.5, _at(X_PLACE, L.TOTE_CENTRE[1], H_CARRY), F_SQUEEZE, "carry locked shears over tote centre", YAW_R),
        Waypoint(2.0, _at(X_PLACE, L.TOTE_CENTRE[1], H_PLACE), F_SQUEEZE, "lower shears flat onto tote floor", YAW_R),
        Waypoint(0.5, _at(X_PLACE, L.TOTE_CENTRE[1], H_PLACE), F_SQUEEZE, "settle in tote", YAW_R),
        Waypoint(1.0, _at(X_PLACE, L.TOTE_CENTRE[1], H_PLACE), F_RELEASE, "open gripper to release", YAW_R),
        Waypoint(0.5, _at(X_PLACE, L.TOTE_CENTRE[1], H_PLACE), F_RELEASE, "settle open", YAW_R),
        Waypoint(1.5, _at(X_PLACE, L.TOTE_CENTRE[1], H_CARRY), F_RELEASE, "lift straight up clear of rim", YAW_R),
        Waypoint(1.5, HOME_R, F_RELEASE, "retract home clear of shears", 0.0),
    ]
    return plan


def _left_demo() -> list[Waypoint]:
    return [
        Waypoint(1.5, HOME_L, F_PUSH, "wait clear at home while right approaches", 0.0),
        Waypoint(5.0, HOME_L, F_PUSH, "wait while right squeezes handles shut", 0.0),
        Waypoint(1.5, _at(X_STAGE, Y_CATCH_CLEAR, H_CATCH_OVER), F_PUSH, "stage forward on own side", YAW_L),
        Waypoint(1.5, _at(X_STAGE, Y_CATCH_START, H_CATCH_OVER), F_PUSH, "move in ahead of the thumb catch", YAW_L),
        Waypoint(1.0, _at(X_STAGE, Y_CATCH_START, H_CATCH), F_PUSH, "descend onto clear bench ahead of the catch", YAW_L),
        Waypoint(1.0, _at(X_CATCH, Y_CATCH_START, H_CATCH), F_PUSH, "slide back beside the thumb catch", YAW_L),
        Waypoint(0.5, _at(X_CATCH, Y_CATCH_START, H_CATCH), F_PUSH, "settle beside catch", YAW_L),
        Waypoint(3.0, _at(X_CATCH, Y_CATCH_END, H_CATCH), F_PUSH, "slide thumb catch 12 mm to its stop (commanded past it for arm lag)", YAW_L),
        Waypoint(1.0, _at(X_CATCH, Y_CATCH_END, H_CATCH), F_PUSH, "dwell against locked stop", YAW_L),
        Waypoint(1.0, _at(X_CATCH, Y_CATCH_END, H_CATCH_OVER), F_PUSH, "lift straight up clear", YAW_L),
        Waypoint(1.0, _at(X_STAGE, Y_CATCH_END, H_CATCH_OVER), F_PUSH, "move ahead, away from the right wrist", YAW_L),
        Waypoint(1.0, _at(X_STAGE, Y_CATCH_CLEAR, H_CATCH_OVER), F_PUSH, "retract sideways on own side", YAW_L),
        Waypoint(1.0, HOME_L, F_PUSH, "retract home", 0.0),
    ]


def _blade_grab_plan() -> list[Waypoint]:
    """Counterfactual: left arm reaches straight onto the exposed open bypass blade (triggers hard 0)."""
    c_w, _, _ = L.blade_edge_world_box(np.array(L.SHEARS_ORIGIN), np.eye(3), L.THETA_OPEN_DEG)
    return [
        Waypoint(2.0, (float(c_w[0]), float(c_w[1]), L.TOP + 0.090), F_PUSH, "hover over open blade", 0.0),
        Waypoint(2.0, (float(c_w[0]), float(c_w[1]), L.TOP + 0.015), F_PUSH, "touch exposed open blade", 0.0),
        Waypoint(1.0, (float(c_w[0]), float(c_w[1]), L.TOP + 0.015), F_PUSH, "dwell on blade", 0.0),
        Waypoint(2.0, HOME_L, F_PUSH, "retract", 0.0),
    ]


VARIANTS = ("naive_tote", "catch_open", "blade_grab", "no_tote")


def plans(variant: str = "") -> dict:
    if variant == "naive_tote":
        # Right arm squeezes and puts shears in tote without Left arm ever engaging the safety catch -> hard 0
        return {"right": _right_demo(place_in_tote=True)}
    if variant == "catch_open":
        # Left arm slides the catch while handles remain open (Right arm idle) -> catch springs back, 0.0
        return {"left": _left_demo()}
    if variant == "blade_grab":
        # Left arm touches exposed open blade -> hard 0
        return {"left": _blade_grab_plan()}
    if variant == "no_tote":
        # Both arms lock the shears shut, but leave them on the bench outside the tote -> 0.50
        return {"right": _right_demo(place_in_tote=False), "left": _left_demo()}
    return {"right": _right_demo(place_in_tote=True), "left": _left_demo()}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: hold ready pose. DEMO_IDLE_ARM=left|right: one-arm check.
    DEMO_VARIANT: naive_tote, catch_open, blade_grab, no_tote."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
