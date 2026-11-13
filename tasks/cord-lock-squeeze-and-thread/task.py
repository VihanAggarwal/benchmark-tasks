"""On the station table, real size: a loose spring cord lock (a Ravenox Extra Large cord lock: 41 x 20 x 13 mm with
its button out, 8.8 g, black Delrin with a stainless spring) lying on one long side, button toward +x, hole's axis
along y, at x 0.302, y 0.130; and 320 mm of loose white 5/16 in. (7.9 mm) shock cord lying straight along y at x 0.300,
one end at y -0.060 pointing at the lock.

The lock's passage is the overlap of its body's hole and its plunger's hole: 4.4 mm wide with the button out, 4.4 mm
plus the travel when pressed, 11.4 x 13 mm with the button flush (7 mm, 10.5 N). The 7.9 mm cord first fits with the
button 3.5 mm in. Goal: press the button, push one cord end through until at least 40 mm stands out of the far side
with at least 40 mm left on the near side, and leave the lock on the table gripping the cord by its spring alone,
both grippers clear. See scoring.py for the rules and the tiers.

The lock is a mechkit assembly (mechkit.py: a body and one sprung slider), the cord a cablekit cable (cablekit.py,
from the program repository's accepted cable tasks: 16 capsules on bending joints), the arms are read through
rigstate.py, and the scripted demo's tilted grips come from tiltgrip.py. None changes what the embodiment does. There
is no rule, weld or bond: the squeeze, the passage and the pinch are contacts and one sprung joint.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) leaves that arm at its ready pose.
Settle:  DEMO_HOLD_ONLY=1 gives no plan (both arms hold the ready pose).
Variants (the README's counterfactuals): DEMO_VARIANT=naive | short | keep_lock | keep_cord | early_lock | in_place |
         slide_right | slide_left.
Trace:   TASK_TRACE=<file> appends one JSON line per scored step.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (import first: it hooks yam_arena so the lock is built as an articulation)
import cablekit  # noqa: E402  (then this: it hooks yam_arena.scene too, so the cord is built as a capsule chain)
import rigstate  # noqa: E402
import tiltgrip  # noqa: E402
from offline_cordlock import OfflineRig  # noqa: E402
from scoring import CORD, LOCK, PLUNGER, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 must hold for 1 s on top of the scorer's own 1 s of release and stillness, so the trial cannot end while a
# gripper holds or touches a part.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Press in the button of the black cord lock and keep it pressed in, with no more "
               "than 2 mm of it standing out, from before the white shock cord's end goes in until "
               "that end is through the lock's hole. Push the end through until at least 4 cm of "
               "cord sticks out of the far side, with at least 4 cm left on the near side. Finish "
               "with the lock resting on the table, gripping the cord by its spring alone, and both "
               "grippers clear of the lock and the cord.")


# -- the cord lock (loose): the body is the base, the plunger slides along -X (pressed in) -----------------------------
def _shapes(table, color):
    return [mk.box(c, s, color) for _, c, s in table]


PLUNGER_LINK = mk.slider(PLUNGER, "-X", _shapes(L.PLUNGER_SHAPES, L.BUTTON_BLACK), lower=0.0, upper=L.TRAVEL,
                         mass=L.PLUNGER_MASS, spring=L.SPRING, damping=L.PLUNGER_DAMPING, friction=0.0,
                         color=L.BUTTON_BLACK)
LOCK_SPEC = mk.assembly(LOCK, origin=(L.LOCK_ORIGIN[0], L.LOCK_ORIGIN[1], L.TOP), mass=L.BODY_MASS,
                        shapes=_shapes(L.BODY_SHAPES, L.BLACK), joints=[PLUNGER_LINK], color=L.BLACK,
                        friction=L.LOCK_FRICTION, tight_fit=True)

# -- the shock cord (loose): end 0 at y -0.060 pointing to +y, end 1 at y -0.380 -----------------------------------------
@dataclasses.dataclass(frozen=True)
class Cord(cablekit.Cable):
    """cablekit's Cable, unchanged in the simulator (cablekit builds any Cable). Only tools/offline/check_task.py calls
    ``offline_box``: there the cord is its world-aligned bounding box, and offline_cordlock.py moves it."""

    def offline_box(self):
        return tuple(float(v) for v in self.pos), tuple(float(v) for v in self.size), 0.0


def _cord(c: cablekit.Cable) -> Cord:
    out = Cord(**{f.name: getattr(c, f.name) for f in dataclasses.fields(c)})
    out._check()
    return out


CORD_SPEC = _cord(cablekit.cable(
    CORD, start=(L.CORD_X, L.CORD_END0_Y, L.TOP + L.CORD_R + L.CORD_SPAWN_LIFT), yaw=-math.pi / 2, length=L.CORD_LEN,
    radius=L.CORD_R, segment=L.CORD_SEGMENT, color=L.WHITE, cable_mass=L.CORD_MASS, bend_stiffness=L.CORD_BEND,
    bend_damping=L.CORD_BEND_DAMPING, collision_sides=L.CORD_SIDES, armature=L.CORD_ARMATURE,
    max_bend=L.CORD_MAX_BEND, friction=L.CORD_FRICTION, joint_kind="d6"))

OBJECTS = [LOCK_SPEC, CORD_SPEC]

# -- the scorer and its hooks ------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineRig()  # only stepped without a simulator (see offline_cordlock.py)


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    OFFLINE.reset()


@mk.on_reset
def _mechkit_reset() -> None:
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the parts: mechkit has no environment AND the lock's state handed in was not
    read from one (mechkit marks those ``source="sim"``). Either sign of a simulator keeps its states untouched."""
    return mk.env() is None and getattr(objects.get(LOCK), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    """Without a simulator nothing presses the button or bends the cord (the offline checker only carries gripped
    boxes), so offline_cordlock.py moves both from the arms' motion; the scorer then sees those states."""
    OFFLINE.step(arms)
    out = dict(objects)
    out[LOCK] = mk.state_from_poses(LOCK_SPEC, (*OFFLINE.origin, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
                                    joints={PLUNGER: OFFLINE.q}, source="offline")
    pts, sd, ed, seg = OFFLINE.cord_arrays()
    lo, hi = OFFLINE.line.min(axis=0) - L.CORD_R, OFFLINE.line.max(axis=0) + L.CORD_R
    out[CORD] = cablekit.CableState(pos=pts.mean(axis=0), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                                    lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=lo, high=hi, points=pts,
                                    start_dir=sd, end_dir=ed, seg_len=seg)
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
    """Through with 40 mm out on both sides, the lock on the table gripping the cord by its spring, both grippers clear
    and everything still for 1 s; held 1 s. This runs every control step, so it also feeds the tracker (and, without a
    simulator, the offline kinematics)."""
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
        row["offline"] = {"lock_holder": OFFLINE.lock_holder, "cord_holder": OFFLINE.cord_holder,
                          "threaded": OFFLINE.threaded, "q_mm": round(OFFLINE.q * 1000, 2)}
    _trace(row)
    return result.success


def score(objects) -> float:
    """Every tier needs the button pressed 5 mm in before a cord end was 5 mm in and held 5 mm in until the cord is
    through (a lost press caps the score at 0.25). 1.0 let go (both grippers clear for
    1 s) with the cord through, 40 mm out on both sides, the lock on the table holding it by its spring (plunger 1 mm or
    more in), all still for 1 s; 0.75 let go and through with 40 mm out; 0.5 let go and through with 10 mm out; 0.25 a
    cord end was 5 mm in at some point; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -------------------------------------------------------------------------------------------------
# Open loop: fixed waypoints from the setup text, no simulator state read. Both grippers close along world x and are
# rolled about world x for the whole demo (tiltgrip.py): the left 0.5 rad with its tips toward -y, the right 0.6 rad
# with its tips toward +y, so the two wrists lean apart (straight down, two YAM grippers this close collide).
# Positions are TCPs (m, table frame); finger targets are minus half the commanded gap.
#
# The left squeezes the lock's ends (the closed end and the button's end face), its TCP 8 mm above the face hole's
# centre, to a 26 mm command: the pads stop on the body's 34 mm length with the button flush (20 N a finger against
# the spring's 10.5 N at the stop), the lock centred between them. It carries the lock 30 mm up to (0.302, 0.040) and
# sets it down there, still squeezed: hole centre at x 0.300, near face at y 0.0335. The right grips the cord 81 mm
# from end 0 (its centre 12 mm below the TCP), raises it to the hole's height (cord at 10.45 mm, the hole's centre at
# 10 mm), brings the end 26.5 mm short of the near face and pushes once along +y, 88 mm at about 15 mm/s: the end comes
# 48.5 mm out of the far face and about 258 mm stays on the near side. The right pad's front corner stops 6 mm short
# of the near face. The left then opens: the plunger follows its pad out until the cord stops it at 3.5 mm in (the
# pinch), and the left rises and moves away; then the right lets go of the cord, rises and moves away.
YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # pads closing along world x
ROLL = {"left": -0.5, "right": 0.6}  # rad about world x
LOCK_OPEN = -0.030  # 60 mm: 9.5 mm clear of each end of the 41 mm lock
SQUEEZE = -0.013  # a 26 mm command across the 34 mm body: 20 N a finger at the stop
CORD_OPEN = -0.015  # 30 mm
CORD_GRIP = 0.0  # closed on the 7.9 mm cord: about 20 N a finger
HOVER_Z = 0.060
LOCK_Z = 0.018  # TCP 8 mm above the face hole's centre; lowest pad corner 3 mm above the table
CARRY_Z = 0.048  # lock 30 mm up
PICK_LOCK = (0.3055, L.LOCK_ORIGIN[1])  # the TCP over the middle of the 41 mm lock
HOLD_LOCK = (0.302, 0.040)
GRIP_BACK = 0.081  # the right grips the cord this far from end 0
CORD_PICK_Z = 0.016  # cord's centre 12 mm below the TCP; lowest pad corner 1 mm above the table
PUSH_Z = 0.0225  # cord at 10.45 mm, the hole's centre at 10 mm
PUSH_START_Y = 0.0335 - 0.0265 - GRIP_BACK  # end 0 26.5 mm short of the near face: -0.074
PUSH_END_Y = 0.014  # end 0 at y 0.095, 48.5 mm past the far face; pad front 6 mm from the near face
PUSH_S = 6.0  # 88 mm in 6 s
UP_Z = 0.080
LOCK_BACK_OFF = 0.030  # an open hand rises leaning back (left +y, right -y), away from the other hand
LEFT_AWAY = (0.300, 0.200, 0.120)
RIGHT_AWAY = (0.300, -0.200, 0.120)
# Per-arm trims (demo only, never the scoring), (dx, dy, dz) m added to every waypoint of that arm after its pick: the
# pads can sit several mm off the commanded point. After a GPU run, read the lock's origin and the cord's end in
# TASK_TRACE: the cord must cross the hole's centre along x within 1.5 mm.
TRIM = {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)}


def _w(arm, duration, xyz, finger, label, trim=True):
    dx, dy, dz = TRIM[arm] if trim else (0.0, 0.0, 0.0)
    pos = (xyz[0] + dx, xyz[1] + dy, L.TOP + xyz[2] + dz)
    return Waypoint(duration, pos, finger, label, tiltgrip.tilted(YAW[arm], ROLL[arm]))


def _variant() -> str:
    v = os.environ.get("DEMO_VARIANT", "")
    known = ("", "naive", "short", "keep_lock", "keep_cord", "early_lock", "in_place", "slide_right", "slide_left")
    if v not in known:
        raise ValueError(f"DEMO_VARIANT={v!r} is not one of {', '.join(k for k in known if k)}")
    return v


def lock_pick(arm: str) -> list[Waypoint]:
    """Down either side of the lock and squeeze it flush (7.5 s)."""
    x, y = PICK_LOCK
    return [
        _w(arm, 2.0, (x, y, HOVER_Z), LOCK_OPEN, "over the lock, fingers 60 mm apart", False),
        _w(arm, 1.0, (x, y, LOCK_Z + 0.030), LOCK_OPEN, "down to 3 cm above the grip", False),
        _w(arm, 1.5, (x, y, LOCK_Z), LOCK_OPEN, "slowly down either side of the lock's ends", False),
        _w(arm, 0.5, (x, y, LOCK_Z), LOCK_OPEN, "settle before the squeeze", False),
        _w(arm, 1.5, (x, y, LOCK_Z), SQUEEZE, "squeeze: the button goes flush, the holes line up", False),
        _w(arm, 0.5, (x, y, LOCK_Z), SQUEEZE, "settle the squeeze", False),
    ]


def lock_carry(arm: str) -> list[Waypoint]:
    """Carry the squeezed lock to where the cord can reach it in one push and set it down, still squeezed (5.5 s)."""
    x, y = PICK_LOCK
    hx, hy = HOLD_LOCK
    return [
        _w(arm, 1.0, (x, y, CARRY_Z), SQUEEZE, "lift the squeezed lock 30 mm"),
        _w(arm, 2.5, (hx, hy, CARRY_Z), SQUEEZE, "carry it in front of the cord"),
        _w(arm, 1.5, (hx, hy, LOCK_Z), SQUEEZE, "slowly down onto the table, still squeezed"),
        _w(arm, 0.5, (hx, hy, LOCK_Z), SQUEEZE, "settle"),
    ]


def lock_let_go(arm: str, at, away) -> list[Waypoint]:
    """Open (the spring pinches the cord), rise and move away (5.0 s)."""
    return [
        _w(arm, 1.5, (at[0], at[1], LOCK_Z), LOCK_OPEN, "open: the plunger springs back onto the cord"),
        _w(arm, 0.5, (at[0], at[1], LOCK_Z), LOCK_OPEN, "settle"),
        _w(arm, 1.0, (at[0], at[1] + LOCK_BACK_OFF, UP_Z), LOCK_OPEN, "up and back, clear of the lock and the other hand"),
        _w(arm, 2.0, away, LOCK_OPEN, "away from the lock"),
    ]


def cord_pick(arm: str, back: float = GRIP_BACK) -> list[Waypoint]:
    """Down either side of the lying cord, 81 mm from its end, and grip it (7.5 s)."""
    y = L.CORD_END0_Y - back
    return [
        _w(arm, 2.5, (L.CORD_X, y, HOVER_Z), CORD_OPEN, "over the cord, fingers 30 mm apart", False),
        _w(arm, 1.0, (L.CORD_X, y, CORD_PICK_Z + 0.030), CORD_OPEN, "down to 3 cm above the cord", False),
        _w(arm, 1.5, (L.CORD_X, y, CORD_PICK_Z), CORD_OPEN, "slowly down either side of the cord", False),
        _w(arm, 0.5, (L.CORD_X, y, CORD_PICK_Z), CORD_OPEN, "settle before the grip", False),
        _w(arm, 1.5, (L.CORD_X, y, CORD_PICK_Z), CORD_GRIP, "grip the cord", False),
        _w(arm, 0.5, (L.CORD_X, y, CORD_PICK_Z), CORD_GRIP, "settle the grip", False),
    ]


def cord_push(arm: str, x: float, start_y: float, end_y: float, back: float = GRIP_BACK) -> list[Waypoint]:
    """Raise the cord to the hole's height, bring its end in front of the near face, wait, push once (11.5 s)."""
    y = L.CORD_END0_Y - back
    return [
        _w(arm, 1.0, (L.CORD_X, y, PUSH_Z), CORD_GRIP, "raise the cord to the hole's height"),
        _w(arm, 3.0, (x, start_y, PUSH_Z), CORD_GRIP, "bring its end 26.5 mm short of the near face"),
        _w(arm, 1.0, (x, start_y, PUSH_Z), CORD_GRIP, "settle: the lock is down and squeezed"),
        _w(arm, PUSH_S, (x, end_y, PUSH_Z), CORD_GRIP, "push once along the hole's axis, about 15 mm/s"),
        _w(arm, 0.5, (x, end_y, PUSH_Z), CORD_GRIP, "settle at the end of the push"),
    ]


def cord_let_go(arm: str, x: float, y: float, away) -> list[Waypoint]:
    """Open, rise and move away (5.0 s)."""
    return [
        _w(arm, 1.5, (x, y, PUSH_Z), CORD_OPEN, "let go of the cord"),
        _w(arm, 0.5, (x, y, PUSH_Z), CORD_OPEN, "settle"),
        _w(arm, 1.0, (x, y - LOCK_BACK_OFF, UP_Z), CORD_OPEN, "up and back, clear of the cord and the other hand"),
        _w(arm, 2.0, away, CORD_OPEN, "away from the cord"),
    ]


def hold(arm: str, seconds: float, plan: list[Waypoint], label: str = "hold") -> list[Waypoint]:
    last = plan[-1]
    return [Waypoint(seconds, last.pos, last.finger, label, last.yaw)]


def _demo_plan() -> dict:
    """Both arms on one clock (29.0 s)."""
    left = lock_pick("left") + lock_carry("left")  # 13.0 s
    right = cord_pick("right") + cord_push("right", L.CORD_X, PUSH_START_Y, PUSH_END_Y)  # 19.0 s
    left += hold("left", 6.0, left, "hold the squeeze while the cord goes through")  # 19.0 s
    left += lock_let_go("left", HOLD_LOCK, LEFT_AWAY)  # 24.0 s
    right += hold("right", 3.0, right, "hold the cord while the lock lets go")  # 22.0 s: the left is up
    right += cord_let_go("right", L.CORD_X, PUSH_END_Y, RIGHT_AWAY)  # 27.0 s
    left += hold("left", 5.0, left, "hold clear")
    right += hold("right", 2.0, right, "hold clear")
    return {"left": left, "right": right}


def _variant_plan(v: str) -> dict:
    """The README's counterfactual runs: each the demo with one change."""
    if v == "naive":  # C1: the right pushes the cord straight at the unpressed lock where it lies
        right = cord_pick("right") + cord_push("right", L.CORD_X, 0.0335 - 0.0265 - GRIP_BACK + 0.090, 0.060)
        right += cord_let_go("right", L.CORD_X, 0.060, RIGHT_AWAY)
        return {"right": right}
    if v == "short":  # C4: the cord gripped 30 mm from its end, one push to the closest pose
        right = cord_pick("right", 0.030) + cord_push("right", L.CORD_X, 0.0335 - 0.0265 - 0.030, PUSH_END_Y, 0.030)
        left = lock_pick("left") + lock_carry("left")
        left += hold("left", right_len(right) - plan_len(left), left)
        left += lock_let_go("left", HOLD_LOCK, LEFT_AWAY)
        right += hold("right", 3.0, right) + cord_let_go("right", L.CORD_X, PUSH_END_Y, RIGHT_AWAY)
        return {"left": left, "right": right}
    if v in ("keep_lock", "keep_cord"):  # C5 / C6: the demo, but one part is never let go
        plan = _demo_plan()
        arm = "left" if v == "keep_lock" else "right"
        n = 11 if arm == "left" else 12  # the waypoints before its let-go
        kept = plan[arm][:n]
        plan[arm] = kept + hold(arm, 29.0 - plan_len(kept), kept, "never let go")
        return plan
    if v == "early_lock":  # C7: the lock set down and let go before the push
        left = lock_pick("left") + lock_carry("left") + lock_let_go("left", HOLD_LOCK, LEFT_AWAY)
        right = cord_pick("right") + cord_push("right", L.CORD_X, PUSH_START_Y, PUSH_END_Y)
        right = right[:7] + hold("right", 18.0 - plan_len(right[:7]), right[:7], "wait for the lock") + right[7:]
        right += cord_let_go("right", L.CORD_X, PUSH_END_Y, RIGHT_AWAY)
        return {"left": left, "right": right}
    if v == "in_place":  # C9: the lock squeezed where it lies; the cord carried to it and pushed in
        x = PICK_LOCK[0] - 0.002  # the hole's centre once the squeezed lock is centred on the TCP
        ny = L.LOCK_ORIGIN[1] - L.HALF_DEPTH
        start, end = ny - 0.0265 - GRIP_BACK, ny - 0.006 - 0.0135
        left = lock_pick("left")
        right = cord_pick("right") + cord_push("right", x, start, end)
        left += hold("left", plan_len(right) - plan_len(left), left)
        left += lock_let_go("left", PICK_LOCK, LEFT_AWAY)
        right += hold("right", 3.0, right) + cord_let_go("right", x, end, RIGHT_AWAY)
        return {"left": left, "right": right}
    if v in ("slide_right", "slide_left"):  # C2 / C3: the squeezed lock slid onto the cord's end
        arm = {"slide_right": "right", "slide_left": "left"}[v]
        x, y = PICK_LOCK
        hx = L.CORD_X + 0.002
        far = L.CORD_END0_Y - 0.045 - L.HALF_DEPTH  # far face 45 mm past end 0 if it threads
        plan = lock_pick(arm) + [
            _w(arm, 1.0, (hx, y, LOCK_Z), SQUEEZE, "line the hole up with the cord"),
            _w(arm, 8.0, (hx, far, LOCK_Z), SQUEEZE, "slide the squeezed lock along the table onto the cord's end"),
            _w(arm, 0.5, (hx, far, LOCK_Z), SQUEEZE, "settle"),
        ] + lock_let_go(arm, (hx, far), LEFT_AWAY if arm == "left" else RIGHT_AWAY)
        out = {arm: plan}
        return out
    raise ValueError(v)


def plan_len(plan) -> float:
    return sum(w.duration for w in plan)


right_len = plan_len


def demo():
    """Both arms at once (29.0 s). DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check). DEMO_VARIANT: see _variant_plan()."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    v = _variant()
    plan = _demo_plan() if not v else _variant_plan(v)
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
