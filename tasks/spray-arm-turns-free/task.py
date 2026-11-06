"""On the station table: a washer rack (a stand-in for a washer-disinfector's lower basket) fixed 27 cm in front of the
arm bases on the midline: a grey floor plate 31.2 cm square with low walls (30 x 30 cm inside), a feed hub at its
centre and, turning freely on the hub, a spray arm (a bar 22 cm long, 85 to 105 mm above the table, with a red cap on
its forward end); two fixed holders, 5 cm tall, fill the rack's far corners. Beside the rack stand two white jars, 12 cm
tall (taller than the arm's underside), and two steel bowls, 4 cm tall, upside down. Loading a washer rack as sterile
processing technicians do: every item standing on the rack floor, the tall ones only where the arm's ends cannot reach
them (5 mm or more outside the circle they sweep: with the holders there, only the two near corners), the short ones
under the arm, then turning the arm by hand half a turn, the load let go of and left undisturbed, to show that it
spins freely.

The spray arm is a mechkit assembly (mechkit.py, next to this file: a fixed-base articulation with one revolute joint
about the vertical); the scorer reads the arm's angle from its link pose (the red end's direction), and the arms (tool
point, finger tips, opening) through rigstate.py, which wraps the embodiment's step and reset from this file's import.
Neither changes what the embodiment does.

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

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena: the spray assembly becomes a mechkit body)
import numpy as np  # noqa: E402
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Held for 1 s with every item still: the 1.0 state already needs both grippers clear of the items (a gripper's
# fingers within 20 mm of an item's side with its finger tips below the item's top score 0), so the hold always ends
# with them let go.
SUCCESS_HOLD_S = L.SUCCESS_HOLD_S

INSTRUCTION = (
    "Load the two white jars and the two steel bowls onto the washer rack, each standing upright on the rack floor "
    "inside its walls, so the spray arm can turn freely: anything taller than the arm's underside must stand at "
    "least 5 mm outside the circle the arm's ends sweep. Then, without moving or touching the load, turn the spray "
    "arm round at least half a turn, so its two ends between them pass over the whole circle. Finish with both "
    "grippers at least 2 cm clear of the items."
)

# -- the washer rack: a fixed floor plate, four fixed walls on its edges and two fixed holders in its far corners ----
RX, RY = L.RACK_XY
_WZ = L.TOP + L.FLOOR_T + L.WALL_H / 2
_HALF = L.FLOOR_SIZE[0] / 2
FLOOR = Box("rack_floor", pos=(RX, RY, L.TOP + L.FLOOR_T / 2), size=L.FLOOR_SIZE, color=L.RACK_GREY)
WALLS = [
    Box("rack_wall_near", pos=(L.INNER_X[0] - L.WALL_T / 2, RY, _WZ), size=(L.WALL_T, L.FLOOR_SIZE[1], L.WALL_H),
        color=L.RACK_GREY),
    Box("rack_wall_far", pos=(L.INNER_X[1] + L.WALL_T / 2, RY, _WZ), size=(L.WALL_T, L.FLOOR_SIZE[1], L.WALL_H),
        color=L.RACK_GREY),
    Box("rack_wall_left", pos=(RX, L.INNER_Y[1] + L.WALL_T / 2, _WZ),
        size=(L.INNER_X[1] - L.INNER_X[0], L.WALL_T, L.WALL_H), color=L.RACK_GREY),
    Box("rack_wall_right", pos=(RX, L.INNER_Y[0] - L.WALL_T / 2, _WZ),
        size=(L.INNER_X[1] - L.INNER_X[0], L.WALL_T, L.WALL_H), color=L.RACK_GREY),
]
HOLDERS = [Box(n, pos=(*xy, L.TOP + L.FLOOR_T + L.HOLDER[2] / 2), size=L.HOLDER, color=L.HOLDER_GREY)
           for n, xy in L.HOLDER_XY.items()]
# -- the feed hub and the spray arm: one fixed assembly, its base frame on the table top at the hub's axis ------------
ARM = mk.hinge(
    "arm", "Z", anchor=(0.0, 0.0, 0.0),
    shapes=[mk.box((0.0, 0.0, L.BAR_Z), L.BAR, color=L.ARM_GREY),
            mk.box((L.CAP_X, 0.0, L.BAR_TOP + L.CAP[2] / 2), L.CAP, color=L.RED)],
    lower=L.ARM_LOWER, upper=L.ARM_UPPER, mass=L.ARM_MASS, damping=L.ARM_DAMPING, color=L.ARM_GREY,
)
SPRAY = mk.assembly("spray", origin=(L.HUB[0], L.HUB[1], L.TOP),
                    shapes=[mk.cylinder((0.0, 0.0, (L.FLOOR_T + L.HUB_TOP) / 2), L.HUB_R, L.HUB_TOP - L.FLOOR_T)],
                    joints=[ARM], color=L.HUB_GREY)
# -- the load: two tall jars and two short bowls, standing on the table beside the rack -----------------------------
JARS = [Cylinder(n, pos=L.START[n], radius=L.JAR_R, height=L.JAR_H, mass=L.JAR_MASS, color=L.WHITE) for n in L.JARS]
BOWLS = [Cylinder(n, pos=L.START[n], radius=L.BOWL_R, height=L.BOWL_H, mass=L.BOWL_MASS, color=L.SILVER)
         for n in L.BOWLS]
OBJECTS = [FLOOR, *WALLS, *HOLDERS, SPRAY, *JARS, *BOWLS]

_TRACKER = S.Tracker()
_MODEL = offline_model.ArmModel()  # offline only (see offline_model.py)
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the load teleported back)."""
    _TRACKER.reset()
    _MODEL.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _arm_yaw(spray, arms, advance: bool) -> float:
    """The spray arm's angle (rad, anticlockwise seen from above): the direction of its red end, from the link's pose
    (never the joint reading). Offline (tools/offline/check_task.py: no physics, the joint stays at 0) the scorer is fed
    offline_model's angle instead: where the grippers would have pushed it. The simulator never takes that branch (its
    states say source "sim"). NaN if the state is broken."""
    if getattr(spray, "source", "sim") == "offline":
        if advance:
            _MODEL.update(arms)
        return _MODEL.theta
    try:
        pose = np.asarray(spray.links["arm"], dtype=float).ravel()
        if pose.size != 7 or not np.all(np.isfinite(pose)) or np.linalg.norm(pose[3:7]) < 0.5:
            raise ValueError("arm pose")
        r = S.rotation(pose[3:7])
        if abs(r[2, 0]) > 0.5:  # the arm's length tilted 30 degrees or more out of level: a broken state
            raise ValueError("arm tilted")
        return math.atan2(r[1, 0], r[0, 0])
    except (AttributeError, KeyError, TypeError, ValueError):
        return math.nan


def _items(objects) -> dict:
    return {n: objects.get(n) for n in L.ITEMS}


def _joint(spray):
    try:
        return round(float(spray.joints["arm"]), 5)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def success(objects) -> bool:
    """Every item loaded, the jars clear of the sweep, both grippers clear of the items and a clean half turn made
    with the load as it stands, let go of; every item still (its path under 5 mm over 0.5 s); held 1 s. This runs
    first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _MODEL.reset()
        _EPISODE["mechkit"] = mk.episode()
    spray = objects.get("spray")
    arms = rigstate.arms()
    yaw = _arm_yaw(spray, arms, advance=True)
    items = _items(objects)
    result = _TRACKER.update(items, yaw, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "arm_yaw_deg": round(math.degrees(yaw), 3) if math.isfinite(yaw) else None,
           "arm_joint": _joint(spray), "source": getattr(spray, "source", None)}
    try:
        row["items"] = {n: [round(float(v), 4) for v in s.pos] for n, s in items.items()}
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tips=arms["left"].tips, right_tips=arms["right"].tips,
                   opening={s: round(float(arms[s].opening), 3) for s in ("left", "right")})
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any item is not standing on the rack floor inside its walls (axis within 10 degrees of vertical, bottom
    within 5 mm of the floor) or a gripper is on an item (its fingers within 20 mm of the item's side, its finger tips
    below the item's top). Else 0.25 if a jar is not clear (its centre less than 14.05 cm from the hub's axis, or its
    side less than 11.55 cm from it at the arm's height); 0.5 if the load stretch running at the end (the load loaded,
    clear, untouched and within 10 mm of where it stood when the stretch began) has not seen the arm turn 180 degrees;
    1.0 otherwise."""
    arms = rigstate.arms()
    yaw = _arm_yaw(objects.get("spray"), arms, advance=False)
    return _TRACKER.result(_items(objects), arms, yaw).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
CLOSED = 0.0  # the pushing gripper: fingers shut, a 13 to 20 mm thick, 27 to 40 mm wide paddle
JAR_GRIP = -(2 * L.JAR_R - 0.008) / 2  # -0.021
BOWL_GRIP = -(2 * L.BOWL_R - 0.008) / 2  # -0.031
YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers closing along x: the signs that reach the midline
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
# TCP heights above the table. Jars: the finger tips 12 mm above the jar's middle (48 mm below its top; the gripper's
# finger carriages, 57 mm above the TCP, stay 22 mm over the jar's top); set down 1 mm above the rack floor.
JAR_PICK = 0.085
JAR_PLACE = L.FLOOR_T + 0.001 + JAR_PICK  # 0.090
JAR_CARRY = 0.150  # a carried jar's bottom 65 mm up: over the walls (24 mm)
JAR_HOVER = 0.160
# Bowls: gripped at their middle (the tips 20 mm up), carried with their bottom 72 mm up (18 mm over the holders' tops,
# which the bowls' paths pass close to), set down 1 mm above the floor; the gripper then rises to the carry height.
BOWL_PICK = L.BOWL_H / 2 + PAD_TIPS  # 0.033
BOWL_PLACE = L.FLOOR_T + 0.001 + BOWL_PICK  # 0.038
BOWL_CARRY = 0.105
SLOW = 0.028  # the last 28 mm of each descent take 1.5 s
CLEAR_UP = 0.135  # over the spray arm: the finger tips 13 mm above its red cap (109 mm)
PUSH_Z = 0.088  # pushing: the finger tips 75 mm up, 10 mm below the arm's underside: the fingers cover its height
PARK = {"left": (0.300, 0.305, L.TOP + 0.200), "right": (0.300, -0.305, L.TOP + 0.200)}  # up, over each base
PUSH1 = (135.0, 240.0)  # phi (degrees about the hub, anticlockwise from forward) of push 1 ...
PUSH2 = (40.0, 240.0)  # ... and push 2
PUSH_STEP_DEG, PUSH_STEP_S = 15.0, 0.4  # waypoints along the circle (47 mm/s mean, 70 mm/s peak)
OVER_STEPS, OVER_S = 6, 0.45  # back round over the arm, from phi 225 to the start of push 2
WAIT_S = 0.5  # the left arm starts its pushes this long after the right arm has parked


def _p(xy, h: float) -> tuple[float, float, float]:
    return (float(xy[0]), float(xy[1]), L.TOP + h)


def _circle(phi_deg: float, h: float) -> tuple[float, float, float]:
    a = math.radians(phi_deg)
    return _p((L.HUB[0] + L.PUSH_R * math.cos(a), L.HUB[1] + L.PUSH_R * math.sin(a)), h)


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _pick_place(side: str, what: str, frm, to, pick: float, place: float, carry: float, hover: float,
                grip: float, first: float) -> list[Waypoint]:
    """Over the item, down beside it (the last 28 mm slowly), settle, grip, lift, carry over the rack, down (the last
    28 mm slowly), settle, let go, up."""
    y = YAW[side]
    return [
        Waypoint(first, _p(frm, hover), OPEN, f"over the {what}", y),
        Waypoint(1.0, _p(frm, pick + SLOW), OPEN, f"down toward the {what}", y),
        Waypoint(1.5, _p(frm, pick), OPEN, f"down beside the {what}, slowly", y),
        Waypoint(0.6, _p(frm, pick), OPEN, "settle", y),
        Waypoint(1.5, _p(frm, pick), grip, f"grip the {what}", y),
        Waypoint(0.3, _p(frm, pick), grip, "settle", y),
        Waypoint(1.2, _p(frm, carry), grip, f"lift the {what}", y),
        Waypoint(1.8, _p(to, carry), grip, f"carry the {what} over its place in the rack", y),
        Waypoint(1.0, _p(to, place + SLOW), grip, "down toward the rack floor", y),
        Waypoint(1.5, _p(to, place), grip, "set it on the rack floor, slowly", y),
        Waypoint(0.6, _p(to, place), grip, "settle", y),
        Waypoint(1.5, _p(to, place), OPEN, f"let go of the {what}", y),
        Waypoint(1.0, _p(to, carry if what.endswith("bowl") else hover), OPEN, f"up, clear of the {what}", y),
    ]


def _jar(side: str, first: float) -> list[Waypoint]:
    n = f"jar_{side}"
    return _pick_place(side, f"{side} jar", L.START[n][:2], L.JAR_PLACE[n], JAR_PICK, JAR_PLACE, JAR_CARRY,
                       JAR_HOVER, JAR_GRIP, first)


def _bowl(side: str, first: float) -> list[Waypoint]:
    n = f"bowl_{side}"
    return _pick_place(side, f"{side} bowl", L.START[n][:2], L.BOWL_PLACE[n], BOWL_PICK, BOWL_PLACE, BOWL_CARRY,
                       BOWL_CARRY, BOWL_GRIP, first)


def _right() -> list[Waypoint]:
    """Right arm: its bowl first (while the left arm sets its jar in the near corner), then its jar (while the left
    arm sets its bowl, 10 cm further out and 21 cm across), then up and out of the way over its base."""
    plan = _bowl("right", 2.2) + _jar("right", 2.0)
    return plan + [Waypoint(2.0, PARK["right"], OPEN, "park, out of the way", YAW["right"])]


T_RIGHT_PARKED = _span(_right())


def _push(a: float, b: float, label: str) -> list[Waypoint]:
    """Along the push circle from phi ``a`` to ``b`` (anticlockwise), every 15 degrees: the shut fingers push the
    arm's end round ahead of them."""
    n = int(round((b - a) / PUSH_STEP_DEG))
    y = YAW["left"]
    return [Waypoint(PUSH_STEP_S, _circle(a + (b - a) * i / n, PUSH_Z), CLOSED,
                     f"{label}: push the spray arm round to {a + (b - a) * i / n:.0f} deg", y) for i in range(1, n + 1)]


def _left() -> list[Waypoint]:
    """Left arm: its jar into the near left corner, its bowl under the arm's sweep beside the hub; then, once the
    right arm has parked, shut the fingers and push the spray arm round in two strokes of 105 and 200 degrees about
    the hub, going back over the arm between them; then up and away."""
    y = YAW["left"]
    plan = _jar("left", 2.2) + _bowl("left", 2.0)
    wait = T_RIGHT_PARKED + WAIT_S - _span(plan)
    plan.append(Waypoint(max(wait, 0.3), plan[-1].pos, OPEN, "wait until the right arm has parked", y))
    a1, b1 = PUSH1
    a2, b2 = PUSH2
    plan += [
        Waypoint(2.0, _circle(a1, CLEAR_UP), CLOSED, "shut the fingers, over the push start", y),
        Waypoint(1.5, _circle(a1, PUSH_Z), CLOSED, "down beside the spray arm, slowly", y),
        Waypoint(0.5, _circle(a1, PUSH_Z), CLOSED, "settle", y),
        *_push(a1, b1, "push 1"),
        Waypoint(0.6, _circle(b1 - 15.0, PUSH_Z), CLOSED, "back off 15 degrees", y),
        Waypoint(1.0, _circle(b1 - 15.0, CLEAR_UP), CLOSED, "up, over the spray arm", y),
    ]
    a0 = b1 - 15.0
    plan += [Waypoint(OVER_S, _circle(a0 + (a2 - a0) * i / OVER_STEPS, CLEAR_UP), CLOSED,
                      "back round over the spray arm", y) for i in range(1, OVER_STEPS + 1)]
    plan += [
        Waypoint(1.5, _circle(a2, PUSH_Z), CLOSED, "down behind the red end, slowly", y),
        Waypoint(0.5, _circle(a2, PUSH_Z), CLOSED, "settle", y),
        *_push(a2, b2, "push 2"),
        Waypoint(0.6, _circle(b2 - 15.0, PUSH_Z), CLOSED, "back off 15 degrees", y),
        Waypoint(1.0, _circle(b2 - 15.0, CLEAR_UP), CLOSED, "up, clear of the spray arm", y),
        Waypoint(2.0, PARK["left"], CLOSED, "park, out of the way", y),
    ]
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
