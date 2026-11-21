"""On the station table, real-size parts of the foot end of a manual hospital bed: a grey steel crank gearbox
(90 x 90 x 160 mm) whose 12 mm shaft projects 130 mm toward the robot, a fold-away crank handle pinned on the shaft's tip
(steel flat 75 mm from pin to knob, red nylon knob 25 mm x 70 mm), and behind the gearbox a head-deck panel on its hinge.

Goal: raise the deck to 30 to 45 degrees by turning the crank (5.0 degrees of deck per full turn: 6 to 9 turns), then
fold the handle back along the shaft and let go. The worm drive is a rule (scoring.Winder): the deck's target angle is
5.0 degrees times the crank's net turns (never below flat), and the crank meets a load while it turns forward. See
scoring.py for the rule and the tiers.

The bed is one mechkit assembly (mechkit.py next to this file); the rule and the crank's whole-turn angle rewrite run in
a mechkit on_step callback in the simulator, and the arms are read through rigstate.py. Neither changes what the
embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
Variants of the demo (counterfactuals): DEMO_VARIANT=short | over | toomuch | unfolded | held (see demo()).
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assembly is built as a compound mechanism)
import rigstate  # noqa: E402
from scoring import Tracker, Winder, crank_target, wrap_shift, wrap_to_pi  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 must hold for 1 s on top of the scorer's own 1 s of release and stillness, so the trial cannot end while a
# gripper holds or touches the crank.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Raise the head section of the manual bed to between 30 and 45 degrees with the foot-end crank: unfold "
               "the handle, turn it forward (clockwise seen from the robot) until the head section is in that range, "
               "then fold the handle back along its shaft and let go.")


# -- the bed ----------------------------------------------------------------------------------------------------------
def _gearbox_and_post() -> list:
    return [
        mk.box(((L.BOX_X[0] + L.BOX_X[1]) / 2, L.SHAFT_Y, (L.BOX_Z[0] + L.BOX_Z[1]) / 2),
               (L.BOX_X[1] - L.BOX_X[0], L.BOX_Y[1] - L.BOX_Y[0], L.BOX_Z[1] - L.BOX_Z[0]), L.GREY),
        mk.box(((L.POST_X[0] + L.POST_X[1]) / 2, (L.POST_Y[0] + L.POST_Y[1]) / 2, (L.POST_Z[0] + L.POST_Z[1]) / 2),
               (L.POST_X[1] - L.POST_X[0], L.POST_Y[1] - L.POST_Y[0], L.POST_Z[1] - L.POST_Z[0]), L.FRAME_GREY),
    ]


class Bed(mk.Assembly):
    """The bed exactly as mechkit builds and reads it in the simulator. Only tools/offline/check_task.py calls the two
    methods below: offline it is the gearbox (without the handle and the deck, which the checker cannot move), and its
    state is rebuilt from the declaration (it is fixed)."""

    def offline_box(self):
        lo = np.array([L.BOX_X[0], L.BOX_Y[0], L.BOX_Z[0]])
        hi = np.array([L.BOX_X[1], L.BOX_Y[1], L.BOX_Z[1]])
        centre = np.asarray(self.pos, dtype=float) + (lo + hi) / 2
        return tuple(float(v) for v in centre), tuple(float(v) for v in hi - lo), float(self.yaw)

    def offline_state(self, state):
        return self.rest_state()


_STUB_C = ((L.STUB_X[0] + L.STUB_X[1]) / 2, L.SHAFT_Y, L.SHAFT_Z)
_PIN_C = (L.PIN_X, L.SHAFT_Y, L.SHAFT_Z)
_ARM_C = (L.PIN_X + L.ARM_LEN / 2, L.SHAFT_Y, L.SHAFT_Z)
_KNOB_C = (L.PIN_X + L.ARM_LEN, L.SHAFT_Y - L.KNOB_OFF, L.SHAFT_Z)  # folded (as drawn): the knob points toward -y
_DECK_ANCHOR = (L.DECK_HINGE_X, L.DECK_Y, L.DECK_HINGE_Z)

_JOINTS = [
    mk.hinge("crank", "X", _STUB_C,
             [mk.cylinder(_STUB_C, L.STUB_R, L.STUB_X[1] - L.STUB_X[0], "X", L.STEEL)],
             lower=-L.CRANK_LIMIT, upper=L.CRANK_LIMIT, mass=L.CRANK_MASS, spring=L.CRANK_K, damping=L.CRANK_DAMP,
             friction=L.BEARING_FRICTION),
    mk.hinge("fold", "-Z", _PIN_C,
             [mk.cylinder(_PIN_C, L.PIN_R, L.PIN_LEN, "Z", L.STEEL),
              mk.box(_ARM_C, (L.ARM_LEN + 2 * L.ARM_BACK, L.ARM_W, L.ARM_T), L.BLACK)],
             lower=0.0, upper=L.FOLD_OPEN, mass=L.ARM_MASS, damping=L.FOLD_DAMP, friction=L.FOLD_FRICTION,
             parent="crank"),
    mk.hinge("knob", "Y", _KNOB_C, [mk.cylinder(_KNOB_C, L.KNOB_R, L.KNOB_LEN, "Y", L.RED)],
             lower=-L.KNOB_LIMIT, upper=L.KNOB_LIMIT, mass=L.KNOB_MASS, damping=L.KNOB_DAMP, friction=L.KNOB_FRICTION,
             parent="fold"),
    mk.hinge("deck", "-Y", _DECK_ANCHOR,
             [mk.box((L.DECK_HINGE_X + L.DECK_LEN / 2, L.DECK_Y, L.DECK_HINGE_Z), (L.DECK_LEN, L.DECK_WID, L.DECK_T),
                     L.DECK_GREY)],
             lower=0.0, upper=L.DECK_LIMIT, mass=L.DECK_MASS, spring=L.DECK_K, damping=L.DECK_DAMP),
]
BED = Bed(name="bed", pos=(0.0, 0.0, L.TOP), color=L.GREY, mass=None, yaw=0.0, shapes=tuple(_gearbox_and_post()),
          joints=tuple(_JOINTS), friction=(0.5, 0.4), **mk.TIGHT_FIT)
BED._check()

OBJECTS = [BED]

# -- the rule: the worm drive, the crank's load and the crank's whole-turn rewrite ------------------------------------
WINDER = Winder()
_WRAP = {"method": None, "failed": None, "writes": 0}


def _write_joint_positions(values: dict) -> None:
    """Rewrite some of the bed's joint positions (rad) in the simulator, velocities unchanged (env 0). Only used for
    whole-turn rewrites of the crank and the knob, which leave every part where it is."""
    import torch

    base = mk.env().unwrapped
    asset = base.scene["bed"]
    names = list(asset.joint_names)
    ids = [names.index(j) for j in values]
    q_all = mk._t(asset.data.joint_pos)
    qd_all = mk._t(asset.data.joint_vel)
    pos = torch.tensor([[float(values[j]) for j in values]], dtype=q_all.dtype, device=q_all.device)
    vel = qd_all[:1, ids].clone()
    attempts = (
        ("write_joint_position_to_sim_index", lambda f: f(position=pos, joint_ids=ids)),
        ("write_joint_position_to_sim", lambda f: f(pos, joint_ids=ids)),
        ("write_joint_state_to_sim_index", lambda f: f(position=pos, velocity=vel, joint_ids=ids)),
        ("write_joint_state_to_sim", lambda f: f(pos, vel, joint_ids=ids)),
    )
    if _WRAP["method"] is not None:
        attempts = tuple(a for a in attempts if a[0] == _WRAP["method"])
    errors = []
    for name, call in attempts:
        fn = getattr(asset, name, None)
        if fn is None:
            continue
        try:
            call(fn)
        except Exception as exc:  # noqa: BLE001 - try the next signature, then report all
            errors.append(f"{name}: {exc!r}")
            continue
        if _WRAP["method"] is None:
            print(f"[bed] whole-turn rewrite uses {name} (joints {list(values)})", file=sys.stderr, flush=True)
        _WRAP["method"] = name
        _WRAP["writes"] += 1
        return
    raise RuntimeError("no joint-position writer worked: " + ("; ".join(errors) or "none of the methods exist"))


@mk.on_step
def _rules(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics): the worm drive,
    then the whole-turn rewrite of the crank and the knob, then the deck's and the crank's drive targets."""
    b = states.get("bed")
    WINDER.update(b)
    try:
        th = float(b.joints["crank"])
        w = float(b.joint_vel.get("crank", 0.0))
        qk = float(b.joints["knob"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    shift, kshift = wrap_shift(th), wrap_shift(qk)
    if (shift or kshift) and _WRAP["failed"] is None:
        values = {}
        if shift:
            values["crank"] = th + shift
        if kshift:
            values["knob"] = qk + kshift
        try:
            _write_joint_positions(values)
        except Exception as exc:  # noqa: BLE001 - loud, once; the crank then stops at its limit (the run shows it)
            _WRAP["failed"] = repr(exc)
            print(f"[bed] ERROR: the whole-turn rewrite failed, the crank will stop at +-{L.CRANK_LIMIT} rad: "
                  f"{exc!r}", file=sys.stderr, flush=True)
            shift = 0.0
    elif _WRAP["failed"] is not None:
        shift = 0.0
    mk.set_targets(BED, "deck", WINDER.target)
    mk.set_targets(BED, "crank", crank_target(th + shift, w, WINDER.drag))


@mk.on_reset
def _rules_reset() -> None:
    WINDER.reset()
    OFFLINE.reset()


# -- offline only: the bed's joints from the arms (tools/offline/check_task.py has no physics) --------------------------
HELD_OPENING = 0.3  # the right gripper closed (opening 0..1) ...
HELD_TOL = 0.025  # ... with its TCP within 25 mm of where the grip point of the knob would be


def _grip_polar_offset() -> float:
    """The grip point's angle behind the arm's direction, seen from the pin (rad): the knob sits KNOB_OFF + GRIP_OFF
    to the arm's right."""
    return math.atan2(L.KNOB_OFF + L.GRIP_OFF, L.ARM_LEN)


class OfflineBed:
    """The offline checker has no joints. This rebuilds the bed's joints from what it plays: while the right gripper
    is closed within HELD_TOL of the knob's grip point, either the crank angle (the knob moves round the shaft) or the
    fold angle (the knob moves round the pin) follows the TCP, whichever fits it better; the deck is at the rule's
    target. Never used in the simulator."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phi = 0.0
        self.fold = 0.0
        self.prev: float | None = None

    @staticmethod
    def _local(tcp) -> np.ndarray:
        t = np.asarray(tcp, dtype=float)
        return np.array([t[0], t[1], t[2] - L.TOP])

    def _residual(self, phi: float, fold: float, p: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(L.grip_point_local(phi, fold)) - p))

    def follow(self, arms) -> None:
        right = arms.get("right") if isinstance(arms, dict) else None
        if right is None or float(right.opening) > HELD_OPENING:
            return
        p = self._local(right.tcp)
        # the crank angle that puts the grip point on the TCP's bearing about the shaft
        g0 = np.asarray(L.grip_point_local(0.0, self.fold))
        phi_a = math.atan2(p[2] - L.SHAFT_Z, p[1] - L.SHAFT_Y) - math.atan2(g0[2] - L.SHAFT_Z, g0[1] - L.SHAFT_Y)
        # the fold angle that puts the grip point on the TCP's bearing about the pin (in the shaft-turned frame)
        c, s = math.cos(self.phi), math.sin(self.phi)
        dy, dz = p[1] - L.SHAFT_Y, p[2] - L.SHAFT_Z
        ly = c * dy + s * dz
        f_b = -math.atan2(ly, p[0] - L.PIN_X) - _grip_polar_offset()
        f_b = min(max(f_b, 0.0), L.FOLD_OPEN)
        ra, rb = self._residual(phi_a, self.fold, p), self._residual(self.phi, f_b, p)
        if min(ra, rb) > HELD_TOL:
            return
        if self.fold >= L.FOLD_OPEN - 0.02 and ra <= rb + 0.004:
            self.phi = phi_a
        elif rb < ra:
            self.fold = f_b
        else:
            self.phi = phi_a

    def state(self, arms, winder: Winder):
        self.follow(arms)
        w = 0.0 if self.prev is None else wrap_to_pi(self.phi - self.prev) / L.DT
        self.prev = self.phi
        origin, quat = BED.spawn_pose()
        tmp = mk.state_from_poses(BED, (*origin, *quat), source="offline",
                                  joints={"crank": wrap_to_pi(self.phi), "fold": self.fold, "deck": winder.target,
                                          "knob": wrap_to_pi(-self.phi)},
                                  joint_vel={"crank": w})
        winder.update(tmp)
        return mk.state_from_poses(BED, (*origin, *quat), source="offline",
                                   joints={"crank": wrap_to_pi(self.phi), "fold": self.fold, "deck": winder.target,
                                           "knob": wrap_to_pi(-self.phi)},
                                   joint_vel={"crank": w})


OFFLINE = OfflineBed()

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a deck that fell back to flat)."""
    _TRACKER.reset()


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
    """Deck at 30 to 45 degrees, handle folded, released and still for 1 s; held 1 s. This runs every control step,
    so it also feeds the tracker (and, offline, the worm-drive rule)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            WINDER.reset()
            OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    if mk.env() is None:  # offline (no simulator, so no on_step): the joints from the arms, then the rule here
        objects = dict(objects)
        try:
            objects["bed"] = OFFLINE.state(arms, WINDER)
        except Exception:  # noqa: BLE001 - keep the checker's own state; the tracker then sees a flat deck
            pass
    result = _TRACKER.update(objects, arms, WINDER)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details, "winder": WINDER.info,
           "wrap": {"method": _WRAP["method"], "writes": _WRAP["writes"], "failed": _WRAP["failed"]}}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    try:
        b = objects.get("bed")
        row["joints"] = {k: round(float(v), 4) for k, v in b.joints.items()}
    except (AttributeError, TypeError, ValueError):
        pass
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 the deck at 30.0 to 45.0 degrees, the handle folded, released and still for 1 s; 0.5 the deck in range but
    not that; 0.25 handle folded and the deck at 15 to 30 or 45 to 60 degrees; 0 otherwise, and 0 once the deck has
    passed 60 degrees. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo (open loop: a fixed schedule of TCP poses and finger targets; it reads no simulator state) ------
# Finger targets: the joint value is minus half the commanded gap. The grip is straight down; the fingers close across
# the knob, so the yaw follows the knob's axis: pi/2 (closing along x) while the handle is folded and the knob points
# along -y, 0 (closing along y) once it is unfolded and the knob points toward the robot.
YAW_FOLDED = math.pi / 2
OPEN_KNOB = -0.0205  # 41 mm apart: 8 mm from each side of the 25 mm knob
GRIP_KNOB = -(2 * L.KNOB_R - 0.008) / 2  # a 17 mm command on the 25 mm knob (about 20 N a finger)
STEP_S = 1.0 / L.CONTROL_HZ
OVER_H = 0.185  # TCP height above the table over the folded knob (70 mm above its centre)
TURNS = 8  # 40 degrees of deck: the middle of the 30 to 45 range, and whole turns end with the pin vertical
STEPS_PER_TURN = 48  # 2.4 s a turn at full speed: the knob at 0.20 m/s
RAMP_STEPS = 20
ORBIT_STEPS = TURNS * STEPS_PER_TURN + RAMP_STEPS
FOLD_STEPS = 40  # 2 s for the 90 degree swing
ORBIT_PULL = 0.012  # the command's circle is 12 mm wider than the knob's: the arm's compliance carries the crank
OVERTRAVEL = 0.03  # rad past the folded stop
RETREAT_X, RETREAT_Y = 0.150, -0.220  # where the hand goes after letting go: 135 mm from the folded knob


def _world(p) -> tuple[float, float, float]:
    return (p[0], p[1], L.TOP + p[2])


def grip_world(phi: float, fold: float) -> tuple[float, float, float]:
    return _world(L.grip_point_local(phi, fold))


def orbit_angle(k: int, turns: int = TURNS) -> float:
    """The commanded crank angle (rad) k control steps into the orbit: speed ramps up linearly over RAMP_STEPS, holds,
    and ramps down over the last RAMP_STEPS; the total is ``turns`` whole turns."""
    total = turns * STEPS_PER_TURN + RAMP_STEPS
    w = 2 * math.pi / STEPS_PER_TURN
    k = min(max(int(k), 0), total)
    r = RAMP_STEPS
    if k <= r:
        return w * k * k / (2 * r)
    if k <= total - r:
        return w * (r / 2 + (k - r))
    m = total - k
    return 2 * math.pi * turns - w * m * m / (2 * r)


def orbit_radius(k: int, turns: int = TURNS) -> float:
    """The commanded circle's radius k steps into the orbit: the knob's own plus ORBIT_PULL, blended in over the
    first ramp and out over the last, so the orbit starts and ends with the TCP on the knob."""
    total = turns * STEPS_PER_TURN + RAMP_STEPS
    k = min(max(int(k), 0), total)
    blend = min(1.0, k / RAMP_STEPS, (total - k) / RAMP_STEPS)
    return L.ARM_LEN + ORBIT_PULL * blend


def orbit_point(k: int, sign: float = 1.0, turns: int = TURNS) -> tuple[float, float, float]:
    """The right TCP's command k control steps into the orbit (``sign`` -1: backward)."""
    phi = sign * orbit_angle(k, turns)
    r = orbit_radius(k, turns)
    x = L.PIN_X - L.KNOB_OFF - L.GRIP_OFF
    return (x, L.SHAFT_Y - r * math.cos(phi), L.TOP + L.SHAFT_Z - r * math.sin(phi))


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


def swing_point(k: int, steps: int, opening: bool, phi: float = 0.0) -> tuple[tuple[float, float, float], float]:
    """The right TCP and yaw k steps into a swing of the handle about its pin (opening: folded to unfolded)."""
    u = _ease(k / steps)
    f = L.FOLD_OPEN * u if opening else L.FOLD_OPEN * (1.0 - u) - OVERTRAVEL * u
    return grip_world(phi, f), YAW_FOLDED - f


def _right(variant: str) -> list[Waypoint]:
    folded = grip_world(0.0, 0.0)
    over = (folded[0], folded[1], L.TOP + OVER_H)
    plan = [
        Waypoint(1.5, over, OPEN_KNOB, "over the folded handle's knob, pointing along -y", YAW_FOLDED),
        Waypoint(1.2, folded, OPEN_KNOB, "down either side of the knob", YAW_FOLDED),
        Waypoint(0.3, folded, OPEN_KNOB, "settle", YAW_FOLDED),
        Waypoint(1.5, folded, GRIP_KNOB, "close on the knob", YAW_FOLDED),
        Waypoint(0.5, folded, GRIP_KNOB, "settle the grip", YAW_FOLDED),
    ]
    for k in range(1, FOLD_STEPS + 1):
        p, yaw = swing_point(k, FOLD_STEPS, True)
        plan.append(Waypoint(STEP_S, p, GRIP_KNOB, "unfold the handle", yaw))
    unfolded = grip_world(0.0, L.FOLD_OPEN)
    plan.append(Waypoint(0.5, unfolded, GRIP_KNOB, "hold the handle out", 0.0))
    turns = {"short": 4, "over": 11, "toomuch": 13}.get(variant, TURNS)  # counterfactuals: 20, 55 and 65 degrees
    for k in range(1, turns * STEPS_PER_TURN + RAMP_STEPS + 1):
        plan.append(Waypoint(STEP_S, orbit_point(k, 1.0, turns), GRIP_KNOB, "turn the crank forward", 0.0))
    plan.append(Waypoint(0.5, unfolded, GRIP_KNOB, "hold the knob where the turn ended", 0.0))
    if variant != "unfolded":
        for k in range(1, FOLD_STEPS + 1):
            p, yaw = swing_point(k, FOLD_STEPS, False)
            plan.append(Waypoint(STEP_S, p, GRIP_KNOB, "fold the handle back along the shaft", yaw))
    end = plan[-1].pos if variant != "unfolded" else unfolded
    end_yaw = YAW_FOLDED - (0.0 if variant != "unfolded" else L.FOLD_OPEN)
    if variant == "held":  # counterfactual: the handle folded but the hand stays closed on the knob
        return plan + [Waypoint(6.0, end, GRIP_KNOB, "keep holding the knob", end_yaw)]
    up = (end[0], end[1], L.TOP + OVER_H)
    back = (RETREAT_X, RETREAT_Y, L.TOP + OVER_H)
    return plan + [
        Waypoint(0.5, end, GRIP_KNOB, "hold", end_yaw),
        Waypoint(1.5, end, OPEN_KNOB, "let go of the knob", end_yaw),
        Waypoint(1.0, up, OPEN_KNOB, "up", end_yaw),
        Waypoint(1.5, back, OPEN_KNOB, "back toward the robot, clear of the shaft, arm and knob", end_yaw),
    ]


def demo():
    """The right arm alone: unfold the handle, turn the crank 8 times, fold the handle away, let go and rise.
    DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's
    plan is dropped (the one-arm check). DEMO_VARIANT (counterfactuals): short (4 turns, 20 degrees), over (11 turns,
    55 degrees), toomuch (13 turns, 65 degrees: past the 60 degree gate), unfolded (the handle left out), held (the
    hand stays closed on the knob)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    variant = os.environ.get("DEMO_VARIANT", "")
    plan = {"right": _right(variant)}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
