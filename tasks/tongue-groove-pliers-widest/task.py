"""On the station bench, real size: one loose pair of Channellock 420 straight-jaw tongue-and-groove pliers (9.5 in,
391 g) lying flat, rivet half up, blue handles toward the robot, set to the narrowest jaw (groove 1 of 5), handles
30 deg apart. Nothing else on the bench and nothing fixed. Three bodies on one loose articulation, laid out as the
real tool is (layout.py): the slotted half (the root, 207.4 g: the left handle and its arm, head plate with its
32.4 mm slot and five arc grooves between six ridges, shoulder, the upper, outer jaw), the rivet (5.2 g) on a slider
along the slot (0 to 24 mm, friction setting 1.5) and the rivet half (178.7 g: the right handle and its bridge, lap
plate, neck, the lower, inner jaw, tongue rib) on a hinge about the rivet (spread 0 to 85 deg). The slot leans 32 deg
off the handle line; groove 1, the narrowest jaw, is its end nearest the jaws; the slide carries the rivet and the
inner jaw back toward the handles (world heading 148 deg, toward the robot and its left), so groove 5, the widest,
is the end nearest the handles.

The groove lock is a rule (``_groove_lock`` below, a mechkit on_step callback): below 72 deg of spread the slider's
spring (10,000 N/m) holds the rivet at the centre of the groove it dropped into, so the joint cannot slide; at 72 deg
and above the spring's target follows the rivet (set each step where the rivet would be if it kept 0.9 of its last
step's travel), so the open joint slides against its friction (setting 1.5) and a light drag. The pliers' own hinge
angle picks the mode. A rib and ridges are
the tongue and the grooves; the halves do not collide with each other (mechkit turns self-collision off inside
an articulation), and no other box of one half overlaps a box of the other at any spread and slide
(layout.interpenetrations).

Goal: open the handles wide, slide the joint to the slot's end nearest the handles (groove 5), close the handles until
it locks, leave the pliers flat with nothing touching them. See scoring.py for the tiers.

The pliers are a mechkit assembly (mechkit.py next to this file); the arms are read through rigstate.py. Neither
changes what the embodiment does.

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
from offline_pliers import OfflinePliers  # noqa: E402
from scoring import PLIERS, SLIP, SPREAD, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of release and stillness

INSTRUCTION = ("The blue-handled tongue-and-groove pliers are set to their narrowest jaw, the first of five grooves. "
               "Reset them to the widest, the fifth groove, and leave them locked there, lying flat on the bench with "
               "nothing touching them.")


# -- the pliers -----------------------------------------------------------------------------------------------------
def _box(centre_mm, size_mm, yaw_deg, color):
    c = tuple(v * L.MM for v in centre_mm)
    s = tuple(v * L.MM for v in size_mm)
    if abs(yaw_deg) < 1e-9:
        return mk.box(c, s, color)
    return mk.box(c, s, color, ypr=(math.radians(yaw_deg), 0.0, 0.0))


def _colour(name: str):
    return L.BLUE if name == "handle" else L.STEEL_GREY


SLOTTED_SHAPES = [_box(c, s, y, _colour(n)) for n, c, s, y in L.slotted_shapes()]
RIVET_SHAPES = [
    mk.cylinder((0.0, 0.0, L.PIN_H / 2 * L.MM), L.PIN_D / 2 * L.MM, L.PIN_H * L.MM, color=L.DARK_STEEL),
    mk.cylinder((0.0, 0.0, (L.PIN_H + L.CAP_H / 2) * L.MM), L.CAP_D / 2 * L.MM, L.CAP_H * L.MM, color=L.DARK_STEEL),
]
RIVET_HALF_SHAPES = [_box(c, s, y, _colour(n)) for n, c, s, y in L.rivet_half_shapes(L.SIGMA0_DEG)]

RIVET = mk.slider(SLIP, L.SLIDE_AXIS, RIVET_SHAPES, anchor=(0.0, 0.0, 0.0), upper=L.SLIDE_TRAVEL, mass=L.RIVET_MASS,
                  spring=L.LOCK_STIFFNESS, damping=L.LOCK_DAMPING, friction=L.SLIDE_FRICTION)
RIVET_HALF = mk.hinge(SPREAD, "Z", (0.0, 0.0, 0.0), RIVET_HALF_SHAPES, lower=L.HINGE_LOWER, upper=L.HINGE_UPPER,
                      mass=L.RIVET_HALF_MASS, damping=L.HINGE_DAMPING, friction=L.HINGE_FRICTION, parent=SLIP)

# Offline the pliers are their head plate's box (no gripper goes near it), and offline_pliers.py moves the rest.
_HEAD_CENTRE = np.array([(L.HEAD_X[0] + L.HEAD_X[1]) / 2 * L.MM, 0.0, 0.006])
_HEAD_SIZE = ((L.HEAD_X[1] - L.HEAD_X[0]) * L.MM, L.HEAD_W * L.MM, 0.012)


class Pliers(mk.Assembly):
    """The pliers exactly as mechkit builds and reads them in the simulator. Only tools/offline/check_task.py calls
    the two methods below: offline the pliers are the head plate's box, which no gripper touches, so the checker's own
    carry proxy leaves them alone; ``offline_state`` puts the pliers' frame back on that box."""

    def offline_box(self):
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ _HEAD_CENTRE
        return tuple(float(v) for v in centre), tuple(float(v) for v in _HEAD_SIZE), float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ _HEAD_CENTRE
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


PLIERS_SPEC = Pliers(name=PLIERS, pos=L.ORIGIN, color=L.STEEL_GREY, mass=L.SLOTTED_MASS, yaw=L.SLOT_YAW,
                     shapes=tuple(SLOTTED_SHAPES), joints=(RIVET, RIVET_HALF), friction=L.FRICTION,
                     solver_iterations=32)
PLIERS_SPEC._check()
OBJECTS = [PLIERS_SPEC]


# -- the groove lock ------------------------------------------------------------------------------------------------
def lock_target(q: float, q_prev: float | None, sigma: float) -> tuple[str, float]:
    """The slider's drive target (m) for one control step, and the mode. ``q_prev``: the slide reading at the previous
    control step (None when there is none to lead from: the first step of a trial, the step after an unreadable one,
    and the first free step after a lock, see ``_groove_lock``).

    Below the release angle the tongue is down in a groove: the target is the nearest groove's centre and the
    10,000 N/m spring holds the rivet there (a rivet left between grooves is pulled into the nearest one, where a
    real tongue could land on a ridge; ``_groove_lock`` then keeps that groove for as long as the joint stays
    locked). At the release angle and above, the target follows the rivet: it is set where the rivet would be at the
    end of this step if it kept FOLLOW_GAIN (0.9) of the travel of its last step, clipped to 1 mm and to the slot.
    The target is written once per 50 ms control step and the spring settles in 25 ms, so a target at the rivet
    itself (or one led by the rivet's instantaneous velocity, which the spring has already stopped) would hold it
    like a 500 N s/m damper; leading by 0.9 of the last step's travel leaves a 50 N s/m drag on top of the slide's
    friction (setting 1.5): a pushed rivet moves, and one that is no longer pushed slows and stops (at 1.0 it coasted
    to the slot's end)."""
    if L.locked(sigma):
        return "locked", L.nearest_groove_target(q)
    step = 0.0 if q_prev is None else L.FOLLOW_GAIN * (q - q_prev)
    lead = float(np.clip(step, -L.FOLLOW_LEAD_MAX, L.FOLLOW_LEAD_MAX))
    return "free", float(np.clip(q + lead, 0.0, L.SLIDE_TRAVEL))


LOCK = {"mode": "locked", "target": 0.0, "switches": 0, "skipped": 0, "q_prev": None}


@mk.on_step
def _groove_lock(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    s = states.get(PLIERS)
    try:
        q = float(s.joints[SLIP])
        sigma = L.sigma_deg(s.joints[SPREAD])
    except (AttributeError, KeyError, TypeError, ValueError):
        q = sigma = math.nan
    if not (math.isfinite(q) and math.isfinite(sigma)):
        LOCK["skipped"] += 1  # an unreadable state: keep the last target, and lead by nothing next step
        LOCK["q_prev"] = None
        mk.set_targets(PLIERS_SPEC, SLIP, LOCK["target"])
        return
    # The lead carries on only travel the joint made while free. On the first free step after a lock the rivet's last
    # move was the lock spring pulling it into its groove. That pull re-opened the gripped handles past 72 deg and the
    # lock let go; led on, the pull carried the joint on toward the next groove (GPU, 13 Nov: a close 15 mm along the
    # slot once ended in groove 5 instead of 4).
    mode, target = lock_target(q, LOCK["q_prev"] if LOCK["mode"] == "free" else None, sigma)
    LOCK["q_prev"] = q
    if mode == "locked" and LOCK["mode"] == "locked":
        target = LOCK["target"]  # the tongue stays in the groove it dropped into: no push walks it to the next one
    if mode != LOCK["mode"]:
        LOCK["switches"] += 1
        print(f"[pliers] groove lock {mode} at episode {mk.episode()} step {mk.step()}: spread {sigma:.2f} deg, "
              f"slide {q * 1000:.2f} mm, target {target * 1000:.2f} mm", file=sys.stderr, flush=True)
    LOCK.update(mode=mode, target=target)
    mk.set_targets(PLIERS_SPEC, SLIP, target)


@mk.on_reset
def _groove_lock_reset() -> None:
    LOCK.update(mode="locked", target=0.0, q_prev=None)
    mk.set_targets(PLIERS_SPEC, SLIP, 0.0)
    OFFLINE.reset()


_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflinePliers()  # only stepped without a simulator (see offline_pliers.py)


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the pliers: mechkit has no environment AND the state handed in was not read
    from one (mechkit marks those ``source="sim"``). Either sign of a simulator keeps its states untouched."""
    return mk.env() is None and getattr(objects.get(PLIERS), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    """Without a simulator nothing moves the pliers' joints, so offline_pliers.py moves them from the arms' commanded
    motion; the scorer then sees those states."""
    OFFLINE.step(arms)
    origin, yaw = OFFLINE.pose()
    out = dict(objects)
    out[PLIERS] = mk.state_from_poses(PLIERS_SPEC, (*origin, *mk.yaw_quat(yaw)), np.zeros(6), joints=OFFLINE.joints(),
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
    """Locked in groove 5, flat, released and still for the scorer's 1 s windows. This runs every control step, so it
    also feeds the tracker (and, without a simulator, the offline kinematics that move the pliers)."""
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
           "fallback_resets": _TRACKER.resets, **result.details,
           "lock": {"mode": LOCK["mode"], "target_mm": round(LOCK["target"] * 1000, 2), "switches": LOCK["switches"],
                    "skipped": LOCK["skipped"]}}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    if offline:
        row["offline"] = {"hold": dict(OFFLINE.hold), "towed_mm": round(OFFLINE.towed * 1000, 1)}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 while any part of the pliers is 5 cm or more below the bench top; 1.0 locked in groove 5, flat, released and
    still; 0.5 the same in groove 2, 3 or 4; 0.2 once the rivet has slid 3 mm out of groove 1 with the joint open;
    0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points (the pad tips
# are 13 mm below). Both grippers point straight down with their fingers closing across a handle (yam_arena's yaw 0
# closes them along world y, so the yaw is the handle's heading less 180 deg). The right gripper takes the rivet half
# 162 mm out along its handle and swings it open about the rivet alone; with the handles 30 deg apart the two grips
# are only 112 mm apart (the wrists' capsules would meet), so the left gripper waits aside until the right has opened
# to 80 deg and only then comes down on the slotted handle, 170 mm out. Then the right carries the rivet along the
# slot to its end nearest the handles (toward the robot and its left, while the left holds the slotted half) and
# eases its push. The slide brings the right gripper toward the left one (the arms stay about 20 mm apart at the
# commanded points), so the left lets go and moves aside before the close; the right then swings the handles in alone
# to 50 deg (the tongue drops into groove 5 at 72), lets go and rises.
# Finger targets: the joint value is minus half the commanded gap.
OPEN = -0.030  # 60 mm: 23.5 mm a side of the 13 mm handle
GRIP = -0.0015  # 3 mm commanded on the 13 mm handle (5 mm of squeeze a side)
H_GRIP = 0.016  # TCP over the bench: the pad tips 3 mm above it, beside the 12 mm handle
H_OVER = 0.080  # right gripper, approach
H_LEFT_OVER = 0.100  # left gripper, approach and park
H_LEFT_LIFT = 0.060
H_LIFT = 0.100
OPEN_TO = 80.0  # deg: the right opens the handles this far (release at 72, stop at 85)
END_SIGMA = 50.0  # deg: the right closes to here alone, 22 deg under the release
OPEN_SPEED = 10.0  # deg/s along the opening arc
CLOSE_SPEED = 5.0  # deg/s along the closing arcs (slow final moves)
CHORD_OPEN, CHORD_CLOSE = 5.0, 2.5  # deg per waypoint along an arc
# The slide is commanded past the stop, because the arms give. Measured on the GPU (13 Nov, this layout): the open
# joint slid with the command about 12 mm ahead of the rivet (the right gripper about 6 mm short of its commanded
# point, the left pulled 1 to 2 mm) and the rivet met the slot's end with the command at 36 mm. 48 mm commanded
# leaves 12 mm over that, and the 2 s dwell lets a slower joint catch up (the tests' 1-D model of the arms' give
# still reaches the stop inside the dwell with half again the friction or a break-away lead of 22 mm; with half the
# arm stiffness it stops short); at the dwell the right gripper stood 10 mm short of its command and the left was
# pulled 4 mm, the handles at 76 deg. Easing back 21 mm leaves the hand 3 mm past where the rivet stops, so the
# closing arc is centred 27 mm along the slot.
SLIDE_CMD = 0.048
EASE_BACK = 0.021
SLIDE_S, DWELL_S, EASE_S = 4.0, 2.0, 1.5
LEFT_PARK = (0.210, 0.113)  # left gripper moves aside here (95 mm to its left) after letting go, while the right closes
HOME = {"left": (0.28, 0.25, 0.12), "right": (0.28, -0.25, 0.12)}  # TCP points (heights over the bench), clear


def _at(xy, h):
    return (float(xy[0]), float(xy[1]), L.TOP + h)


def _yaw_r(sigma):
    return math.radians(L.rivet_grip_yaw_deg(sigma))


YAW_L = math.radians(L.slotted_grip_yaw_deg())


def _arc(q_centre: float, s0: float, s1: float, speed: float, chord: float, label: str) -> list[Waypoint]:
    """The right gripper along the rivet half's grip circle about a rivet ``q_centre`` m along the slot, from spread
    s0 to s1 (deg), in chords of at most ``chord`` deg at ``speed`` deg/s."""
    n = max(1, int(math.ceil(abs(s1 - s0) / chord - 1e-9)))
    out = []
    for k in range(1, n + 1):
        sig = s0 + (s1 - s0) * k / n
        out.append(Waypoint(abs(s1 - s0) / n / speed, _at(L.rivet_grip(sig, q_centre), H_GRIP), GRIP,
                            f"{label} ({sig:.1f} deg)", _yaw_r(sig)))
    return out


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _right_plan(open_to=OPEN_TO, slide=SLIDE_CMD, ease=EASE_BACK, two_hands=True,
                stop_after_slide=False) -> tuple[list[Waypoint], dict]:
    """The right gripper (the rivet half). Returns the plan and the times (s) the left gripper's plan keys on.
    ``stop_after_slide``: let go once the slide is eased, with the handles still open past the release angle."""
    a = L.rivet_grip(L.SIGMA0_DEG)
    y30 = _yaw_r(L.SIGMA0_DEG)
    plan = [
        Waypoint(2.5, _at(a, H_OVER), OPEN, "over the rivet half's handle, 12 mm from its end, open 60 mm", y30),
        Waypoint(1.5, _at(a, H_GRIP), OPEN, "slowly down beside the handle", y30),
        Waypoint(0.5, _at(a, H_GRIP), OPEN, "settle", y30),
        Waypoint(1.5, _at(a, H_GRIP), GRIP, "close on the handle (commanded 3 mm)", y30),
        Waypoint(0.5, _at(a, H_GRIP), GRIP, "settle the grip", y30),
    ]
    plan += _arc(0.0, L.SIGMA0_DEG, open_to, OPEN_SPEED, CHORD_OPEN, "swing the handles open about the rivet")
    times = {"opened": _duration(plan)}
    yo = _yaw_r(open_to)
    top = L.rivet_grip(open_to, 0.0)
    if not two_hands:
        return plan, times
    left_down = times["opened"] + LEFT_DOWN_S
    plan.append(Waypoint(left_down - times["opened"], _at(top, H_GRIP), GRIP,
                         "hold open while the left comes down", yo))
    plan += [
        Waypoint(SLIDE_S, _at(L.rivet_grip(open_to, slide), H_GRIP), GRIP,
                 f"carry the rivet along the slot toward the handles (commanded {slide * 1000:.0f} mm)", yo),
        Waypoint(DWELL_S, _at(L.rivet_grip(open_to, slide), H_GRIP), GRIP, "dwell against the slot's end", yo),
        Waypoint(EASE_S, _at(L.rivet_grip(open_to, slide - ease), H_GRIP), GRIP,
                 f"ease the push (back {ease * 1000:.0f} mm)", yo),
        Waypoint(0.5, _at(L.rivet_grip(open_to, slide - ease), H_GRIP), GRIP, "settle", yo),
    ]
    centre = slide - ease
    if stop_after_slide:
        times["released"] = _duration(plan)
        end = plan[-1].pos
        plan += [
            Waypoint(1.5, end, OPEN, "let go with the handles still open (open to 60 mm)", yo),
            Waypoint(0.5, end, OPEN, "settle, open", yo),
            Waypoint(1.5, (end[0], end[1], L.TOP + H_LIFT), OPEN, "straight up, clear of the pliers", yo),
            Waypoint(2.5, _at(HOME["right"][:2], HOME["right"][2]), OPEN, "home", yo),
        ]
        return plan, times
    times["slid"] = _duration(plan)
    plan.append(Waypoint(LEFT_AWAY_S, plan[-1].pos, GRIP, "hold open while the left lets go and moves aside", yo))
    plan += _arc(centre, open_to, END_SIGMA, CLOSE_SPEED, CHORD_CLOSE, "swing the handles in alone")
    times["closed"] = _duration(plan)
    end = plan[-1].pos
    ye = _yaw_r(END_SIGMA)
    plan += [
        Waypoint(0.5, end, GRIP, "settle", ye),
        Waypoint(1.5, end, OPEN, "let go (open to 60 mm)", ye),
        Waypoint(0.5, end, OPEN, "settle, open", ye),
        Waypoint(1.5, (end[0], end[1], L.TOP + H_LIFT), OPEN, "straight up, clear of the pliers", ye),
        Waypoint(2.5, _at(HOME["right"][:2], HOME["right"][2]), OPEN, "home", ye),
    ]
    return plan, times


LEFT_OVER_S = 2.5  # from its waiting point to over the slotted handle
LEFT_DOWN_S = LEFT_OVER_S + 4.0  # from the right reaching OPEN_TO: over the slotted handle, down 1.5 s, settle 0.5,
# close 1.5, settle 0.5
LEFT_AWAY_S = 5.0  # from the end of the slide: open 1.5, settle 0.5, up 1.5, aside 1.5


def _left_plan(opened: float, slid: float) -> list[Waypoint]:
    """The left gripper (the slotted half): wait aside, come down once the right has opened, hold through the slide,
    let go, move aside and go home."""
    g = L.slotted_grip()
    return [
        Waypoint(2.5, _at(HOME["left"][:2], HOME["left"][2]), OPEN, "wait clear of the right arm, open 60 mm", YAW_L),
        Waypoint(opened - 2.5, _at(HOME["left"][:2], HOME["left"][2]), OPEN, "wait while the right opens the handles",
                 YAW_L),
        Waypoint(LEFT_OVER_S, _at(g, H_LEFT_OVER), OPEN, "over the slotted handle, 15 mm from its end", YAW_L),
        Waypoint(1.5, _at(g, H_GRIP), OPEN, "slowly down beside the handle", YAW_L),
        Waypoint(0.5, _at(g, H_GRIP), OPEN, "settle", YAW_L),
        Waypoint(1.5, _at(g, H_GRIP), GRIP, "close on the handle (commanded 3 mm)", YAW_L),
        Waypoint(0.5, _at(g, H_GRIP), GRIP, "settle the grip", YAW_L),
        Waypoint(slid - opened - LEFT_DOWN_S, _at(g, H_GRIP), GRIP, "hold through the slide", YAW_L),
        Waypoint(1.5, _at(g, H_GRIP), OPEN, "let go (open to 60 mm)", YAW_L),
        Waypoint(0.5, _at(g, H_GRIP), OPEN, "settle, open", YAW_L),
        Waypoint(1.5, _at(g, H_LEFT_LIFT), OPEN, "straight up", YAW_L),
        Waypoint(1.5, _at(LEFT_PARK, H_LEFT_LIFT), OPEN, "aside, clear of the right's closing swing", YAW_L),
        Waypoint(2.5, _at(HOME["left"][:2], HOME["left"][2]), OPEN, "home", YAW_L),
    ]


def _naive_plan() -> list[Waypoint]:
    """Counterfactual: the right gripper takes the rivet half at the 30 deg start and pushes it 45 mm along the slot,
    without opening the handles."""
    a = L.rivet_grip(L.SIGMA0_DEG)
    y30 = _yaw_r(L.SIGMA0_DEG)
    b = L.rivet_grip(L.SIGMA0_DEG, 0.045)
    return [
        Waypoint(2.5, _at(a, H_OVER), OPEN, "over the rivet half's handle", y30),
        Waypoint(1.5, _at(a, H_GRIP), OPEN, "down beside the handle", y30),
        Waypoint(0.5, _at(a, H_GRIP), OPEN, "settle", y30),
        Waypoint(1.5, _at(a, H_GRIP), GRIP, "close on the handle", y30),
        Waypoint(0.5, _at(a, H_GRIP), GRIP, "settle the grip", y30),
        Waypoint(4.0, _at(b, H_GRIP), GRIP, "push it 45 mm along the slot", y30),
        Waypoint(1.0, _at(b, H_GRIP), GRIP, "dwell", y30),
        Waypoint(1.5, _at(b, H_GRIP), OPEN, "let go", y30),
        Waypoint(1.5, _at(b, H_LIFT), OPEN, "straight up", y30),
        Waypoint(2.5, _at(HOME["right"][:2], HOME["right"][2]), OPEN, "home", y30),
    ]


VARIANTS = ("naive", "no_slide", "early_close", "slide_open")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant."""
    if variant == "naive":
        return {"right": _naive_plan()}
    kw = {}
    if variant == "no_slide":
        kw.update(slide=0.0, ease=0.0)
    elif variant == "early_close":
        kw.update(slide=0.024, ease=0.006)
    elif variant == "slide_open":
        right, times = _right_plan(stop_after_slide=True)
        return {"right": right, "left": _left_plan(times["opened"], times["released"])}
    right, times = _right_plan(**kw)
    return {"right": right, "left": _left_plan(times["opened"], times["slid"])}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm checks).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      naive        the right arm alone takes the rivet half at the start and pushes it 45 mm along the slot
      no_slide     the demo without the slide: open to 80 deg, hold both halves, then close again
      early_close  the demo with the slide commanded 24 mm instead of 48, eased back 6 mm (the arms' give takes up
                   part of the command)
      slide_open   the demo up to the end of the slide, then both grippers let go with the handles still open past
                   72 deg and leave (the joint is left open, not locked)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
