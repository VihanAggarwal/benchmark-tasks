"""On the station table: a black rubber crucible former (an 88 x 88 x 10 mm block with a round centring step, 36 mm
across and 6 mm tall, on top) fixed 30 cm in front of the arm bases and 8 cm left of the midline; on the step, 6 mm
left of the former's axis, a red wax sprue (a 4 mm rod, 10 mm tall) carrying a blue wax crown pattern
(10 x 12 x 10 mm, its top 36 mm above the table); and three loose grey stainless-steel casting rings (square tubes
open at both ends, 60 x 60 mm outside with 4 mm walls, a 52 x 52 mm opening) standing on the table in front of the
left arm: ring_a 40 mm tall, ring_b 24 mm, ring_c 32 mm. Stood on the former's base, ring_c's top is 6 mm above the
crown's top (ring_b's 2 mm below it, ring_a's 14 mm above): the only one in the 5-8 mm band. Rule: the chosen ring
goes straight down over the wax and stands level on the former round the step, released; neither a ring nor a
gripper may ever touch the crown or the sprue, and only one ring may end up on the former.

The rings are mechkit assemblies (mechkit.py, next to this file: one loose rigid body made of four walls each); the
scorer reads the arms (tool point, finger tips, opening) through rigstate.py and each gripper's turn about its axis
(its `gripper` link's pose) through gripturn.py, which both wrap the embodiment's step and reset from this file's
import. None of them changes what the embodiment does. offline_model.py is used only by our offline checker (no
physics there): the simulator never calls it.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the demo uses the left arm only; DEMO_IDLE_ARM=left (or right) leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so each ring is built as one compound body)
import gripturn  # noqa: E402
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

_RIG_HOOKED = rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it)
if not _RIG_HOOKED and not getattr(sys.modules.get("yam_arena"), "OFFLINE_STUB", False):
    print("[casting-ring-over-wax] WARNING: rigstate could not hook yam_arena's embodiment, so the grippers cannot "
          "be read: the touch rule cannot be judged and, past 1% of the steps, the trial scores 0", file=sys.stderr)
_TURN_HOOKED = gripturn.install()  # each gripper's link pose, read at the same moment (False offline: turn unknown)

SCENE = "station"
SCORE_OVER = "end"
# Seated, let go and still for 2.5 s. The fingers pass 88 mm apart (no longer holding) about 1.2 s into the 1.5 s
# release, so the trial ends about 3.7 s after the release begins, with the gripper fully open and moving away.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Three steel casting rings stand on the left of the table. Pick the one whose top edge will be 5 to 8 mm above "
    "the top of the blue wax crown once the ring stands on the black former. Lower it straight down over the crown "
    "and stand it on the former around the raised step, then let go. Leave the other two rings off the former. "
    "Never let a ring or a gripper touch the wax crown or its red sprue."
)

# -- the objects --------------------------------------------------------------------------------------------------
FORMER_BASE = Box("former_base", pos=(L.FORMER_X, L.FORMER_Y, L.TOP + L.BASE_SIZE[2] / 2), size=L.BASE_SIZE,
                  color=L.RUBBER_BLACK)
FORMER_STEP = Cylinder("former_step", pos=(L.FORMER_X, L.FORMER_Y, L.BASE_TOP + L.STEP_HEIGHT / 2),
                       radius=L.STEP_RADIUS, height=L.STEP_HEIGHT, color=L.RUBBER_BLACK)
SPRUE = Cylinder("sprue", pos=L.SPRUE_CENTRE, radius=L.SPRUE_RADIUS, height=L.SPRUE_HEIGHT, color=L.SPRUE_RED)
CROWN = Box("wax_crown", pos=L.CROWN_CENTRE, size=L.CROWN_SIZE, color=L.WAX_BLUE)


def _ring_shapes(h: float) -> list:
    """Four 4 mm walls round a 52 x 52 mm opening, in the ring's base frame (the centre of its bottom face)."""
    off = L.RING_OUTER_HALF - L.RING_WALL / 2  # 0.028
    return [
        mk.box((off, 0.0, h / 2), (L.RING_WALL, L.RING_OUTER, h)),
        mk.box((-off, 0.0, h / 2), (L.RING_WALL, L.RING_OUTER, h)),
        mk.box((0.0, off, h / 2), (L.RING_INNER, L.RING_WALL, h)),
        mk.box((0.0, -off, h / 2), (L.RING_INNER, L.RING_WALL, h)),
    ]


RINGS = [
    mk.assembly(name, origin=(x, y, L.TOP), mass=m, color=L.STEEL, shapes=_ring_shapes(h))
    for name, (h, m, x, y) in L.RINGS.items()
]
OBJECTS = [FORMER_BASE, FORMER_STEP, SPRUE, CROWN, *RINGS]

_TRACKER = S.Tracker()
_MODEL = offline_model.RingModel()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
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


def _observe(objects):
    """The rings and the arms the scorer judges this step. In the simulator: the rings' states and rigstate's
    snapshot, with each gripper's turn from gripturn's reading of its link. Offline (tools/offline/check_task.py: no
    physics, and its states say source "offline") the rings come from offline_model, which lets a released ring slide
    down round the wax as a tube does, and the turn is not known. The simulator never takes that branch (its states
    say source "sim")."""
    get = objects.get if hasattr(objects, "get") else (lambda k: None)
    rings = {n: get(n) for n in L.RING_NAMES}
    arms = rigstate.arms()
    offline = any(getattr(r, "source", "sim") == "offline" for r in rings.values()) and all(
        r is not None for r in rings.values())
    if offline:
        rings, arms = _MODEL.step(rings, arms)
    return rings, S.with_turn(arms, gripturn.bodies()), offline


def success(objects) -> bool:
    """ring_c standing level on the former's base round the step, over the wax, let go and still; no touch of the
    wax at any time; no second ring on the former; held 2.5 s. This runs first every control step, so it also feeds
    the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _MODEL.reset()
        _EPISODE["mechkit"] = mk.episode()
    rings, arms, offline = _observe(objects)
    result = _TRACKER.update(rings, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "offline_model": offline}
    if S.arms_valid(arms):
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips)
        for side in S.ARMS:
            r = S.turn_of(arms[side])
            if r is not None:
                row[f"{side}_closing_axis"] = [round(float(v), 4) for v in r[:, 0]]
    else:
        row["rigstate_error"] = rigstate.error()
    if gripturn.error():
        row["gripturn_error"] = gripturn.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if at any moment a ring wall came within 0.5 mm of the wax crown or the sprue, or a crown or sprue point
    was inside a finger (its box grown 1 mm; the box swept round the gripper's axis when its turn is not known); or
    if at the end two or more rings are on the former; or if the arms could not be read on more than 1% of the steps.
    Else 1.0 ring_c seated (over the wax, its four lower outer corners 9-12 mm up, not held, still); 0.5 ring_c over
    the wax but not seated; 0.25 ring_a or ring_b seated; 0 otherwise."""
    rings, arms, _ = _observe(objects)
    return _TRACKER.result(rings, arms).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Left arm only, yaw 0 throughout (fingers closing along y, across the ring's 60 mm outside). The fingers are
# commanded 8 mm narrower than the ring (about 20 N per finger), with the TCP 3 mm below the ring's top so the pads
# hold the top 16 mm of the two walls. The ring is lifted 9 cm clear of everything, carried over the former,
# lowered to 10 mm above the crown's top, settled, then lowered slowly round the wax until its bottom is 1 mm above
# the base, settled and released. Centred on the former's axis its inner walls clear the crown by 14 mm (+y) and
# 21 mm (x), and the step by 8 mm on every side, so the ring may land up to 8 mm off the axis along x or y and still
# stand on the base (and 8 mm off toward -y its +y wall still clears the crown by 6 mm). The fingers close along y,
# so the grip centres the ring in y but not in x: a pad offset at the pick carries into x at the seat. Payload sag: the arm's controller does not carry the ring's 2.3 N. The static estimate
# (J K^-1 J^T of the weight; joints 80/80/80/10/10/10 N m/rad) at the seat is (-2.3, +1.8, -5.2) mm and under 1 deg
# of tilt. The vertical part makes the ring come to rest on the base about 4 mm before the slow descent ends
# (harmless: the base takes the weight and the tilt rocks out). The sideways part would set it down 2.9 mm off the
# axis; PLACE_TRIM moves the place point by half of it, so the ring lands within about 1.5 mm of the axis whether the
# arm sags as estimated or not at all (the offline check). On the GPU read the rings.ring_c fields in TASK_TRACE
# (axis_offset_mm, low/high_corner_mm, tilt_deg, wall_wax_gap_mm) at the release and tune only PLACE_TRIM and
# SEAT_GAP.
OPEN = -0.04695
GRIP = -(L.RING_OUTER - 0.008) / 2  # -0.026: fingers commanded 52 mm apart on the 60 mm ring
YAW = 0.0
GRIP_DOWN = 0.003  # the TCP this far below the ring's top while gripping (pad tips 16 mm down its walls)
CARRY_Z = L.TOP + 0.120  # TCP; the carried ring's bottom is then about 9 cm up
SEAT_GAP = 0.001  # release with the ring's bottom this far above the base's top
ENTRY_GAP = 0.010  # the slow part starts with the ring's bottom this far above the crown's top
PLACE_TRIM = (0.0012, -0.0009)  # added to the former's axis at the place point: half the estimated sag
AWAY = (0.300, 0.260)  # where the left arm ends, 16 cm up, in front of its own base


def ring_plan(name: str = L.RIGHT_RING, place=None, release: bool = True) -> list[Waypoint]:
    """Pick ``name`` at its start, carry it over ``place`` (default: the former's axis plus PLACE_TRIM), lower it
    round the wax onto the base and, with ``release``, let go and go up (the offline tests use the other rings and
    places to reach each tier)."""
    h, _, sx, sy = L.RINGS[name]
    px, py = place if place is not None else (L.FORMER_X + PLACE_TRIM[0], L.FORMER_Y + PLACE_TRIM[1])
    hold = h - GRIP_DOWN  # TCP above the ring's bottom while gripping
    pick_z = L.TOP + hold
    entry_z = L.CROWN_TOP + ENTRY_GAP + hold
    seat_z = L.BASE_TOP + SEAT_GAP + hold
    plan = [
        Waypoint(3.0, (sx, sy, CARRY_Z), OPEN, f"over {name}, open", YAW),
        Waypoint(1.0, (sx, sy, L.TOP + 0.060), OPEN, "down to 6 cm", YAW),
        Waypoint(1.5, (sx, sy, pick_z), OPEN, "down beside its walls, slowly", YAW),
        Waypoint(0.5, (sx, sy, pick_z), OPEN, "settle", YAW),
        Waypoint(1.5, (sx, sy, pick_z), GRIP, "grip it across its outside", YAW),
        Waypoint(1.5, (sx, sy, CARRY_Z), GRIP, "lift it straight up", YAW),
        Waypoint(2.5, (px, py, CARRY_Z), GRIP, "carry it over the former", YAW),
        Waypoint(1.5, (px, py, entry_z), GRIP, "down: its bottom 10 mm over the crown's top", YAW),
        Waypoint(0.5, (px, py, entry_z), GRIP, "settle", YAW),
        Waypoint(2.5, (px, py, seat_z), GRIP, "lower it round the wax, slowly, to 1 mm over the base", YAW),
        Waypoint(0.8, (px, py, seat_z), GRIP, "settle", YAW),
    ]
    if release:
        plan += [
            Waypoint(1.5, (px, py, seat_z), OPEN, "let go", YAW),
            Waypoint(1.5, (px, py, CARRY_Z), OPEN, "straight up", YAW),
            Waypoint(2.0, (*AWAY, L.TOP + 0.160), OPEN, "back toward its own side, clear of the camera's view", YAW),
        ]
    else:
        plan.append(Waypoint(6.0, (px, py, seat_z), GRIP, "keep holding it", YAW))
    return plan


def demo():
    """Left arm only: the right arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose,
    a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": ring_plan()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
