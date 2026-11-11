"""On the station table: a bench model, about half size, of the two rear duo-servo drum brakes of one car. Two fixed
backing plates (120 mm steel discs) lie shoe side up, 34 cm apart on the midline's two sides, each with its anchor
post on the far side (+x), its adjuster block on the near side (-x) and two painted yellow seats, one on each side of
the line between them. A white arrow on each plate, and a larger one on the bench beyond it, points toward the front
of the car: toward +y on the left rear brake, toward -y on the right rear brake (mirror images). Four loose shoes
(curved bars 20 mm wide and 15 mm tall, 0.04 kg, a dark brown lining band along the outer edge of each top) lie
in a row in front of the plates: secondary, primary, primary, secondary from -y; a primary's lining runs over 90
degrees of its arc, a secondary's over 130. Rule: on each plate the primary (shorter lining) goes on the seat toward
the front of the car and the secondary (longer lining) on the rear seat, each shoe flat and let go.

The shoes and the plates are mechkit assemblies (mechkit.py, next to this file): each shoe one loose rigid body of a
block and twelve discs with its lining discs on top, each plate one fixed body with its painted seats and arrows; the
scorer reads each shoe's frame (its mid-arc point and heading). It also reads the arms (pad-tip points, openings)
through rigstate.py, which wraps the embodiment's step and reset from this file's import, only to check that each
shoe was let go. Neither changes what the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the shoes and plates are built as compounds)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 must hold for 1 s. It already needs every shoe let go (no gripper's fingers within reach of it below 10 mm over
# its top) at every step of the last 0.5 s and still over the last 1 s, so the trial cannot end while a gripper
# still holds or touches a shoe.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "These are the two rear drum brakes from one car, laid shoe side up on the bench with each anchor post on the "
    "far side. The white arrow on each backing plate points toward the front of the car from that brake. They are "
    "duo-servo brakes. Fit one primary and one secondary shoe to each backing plate, each shoe lying flat on a "
    "painted seat between the anchor post and the adjuster block, the way they go on this kind of brake, then let "
    "go of them."
)


# -- the shoes (loose compounds) -----------------------------------------------------------------------------------
def shoe_shapes(kind: str):
    """The shoe in its body frame: the grip block at mid-arc, twelve 20 mm discs along its centre line, and its
    lining (6 mm discs, 1 mm thick, on its top along the outer edge)."""
    z = L.SHOE_H / 2
    shapes = [mk.box((0.0, 0.0, z), L.GRIP_BLOCK, color=L.SHOE_GREY)]
    shapes += [mk.cylinder((x, y, z), L.SHOE_HALF_W, L.SHOE_H, color=L.SHOE_GREY) for x, y in L.shoe_disc_centres()]
    shapes += [mk.cylinder((x, y, L.SHOE_H + L.LINING_H / 2), L.LINING_DOT_R, L.LINING_H, color=L.LINING_BROWN)
               for x, y in L.lining_centres(kind)]
    return shapes


_GRIP_CENTRE = np.array([0.0, 0.0, L.SHOE_TOP / 2])
_GRIP_SIZE = (L.GRIP_BLOCK[0], L.GRIP_BLOCK[1], L.SHOE_TOP)


class Shoe(mk.Assembly):
    """A shoe exactly as mechkit builds and reads it in the simulator. Only tools/offline/check_task.py calls the two
    methods below: offline the shoe is its grip block (20 mm across its width at mid-arc, 16 mm along it, 16 mm tall
    with the lining), the part a gripper closes on, so the checker grasps it there, carries it turned with the
    gripper and drops it; ``offline_state`` rebuilds the shoe's frame from that block for the hooks. (The whole shoe's
    box would put its centre 12 mm inside the arc, and the checker centres a grasped box between the pads.)"""

    def offline_box(self):
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ _GRIP_CENTRE
        return tuple(float(v) for v in centre), _GRIP_SIZE, float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ _GRIP_CENTRE
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


def _shoe(name: str) -> Shoe:
    kind, x, y, yaw = L.SHOES[name]
    s = Shoe(name=name, pos=(x, y, L.TOP), color=L.SHOE_GREY, mass=L.SHOE_MASS, yaw=math.radians(yaw),
             shapes=tuple(shoe_shapes(kind)))
    s._check()
    return s


SHOES = [_shoe(n) for n in L.NAMES]
SHOE_NAMES = tuple(s.name for s in SHOES)


# -- the backing plates (fixed compounds), with their anchor posts and adjuster blocks (fixed primitives) ------------
def seat_paint(front: float, side: str):
    """A painted seat in its plate's frame: the footprint of a shoe in it, 0.5 mm proud of the plate."""
    sign = front if side == "front" else -front
    mid = math.radians(90.0 * sign)
    z = L.PLATE_H + L.PAINT_H / 2
    shapes = [mk.box((0.0, sign * L.R_MID, z), (L.GRIP_BLOCK[1], L.GRIP_BLOCK[0], L.PAINT_H), color=L.SEAT_YELLOW)]
    for d in L.DISC_DEGS:
        for s in (-1.0, 1.0):
            a = mid + math.radians(s * d)
            shapes.append(mk.cylinder((L.R_MID * math.cos(a), L.R_MID * math.sin(a), z), L.SHOE_HALF_W, L.PAINT_H,
                                      color=L.SEAT_YELLOW))
    return shapes


def arrow(front: float, parts, dx: float, z0: float):
    """A painted arrow along y toward the front (``front`` +1: +y), made of bars (start, end, width)."""
    return [mk.box((dx, front * (a + b) / 2, z0 + L.PAINT_H / 2), (w, b - a, L.PAINT_H), color=L.ARROW_WHITE)
            for a, b, w in parts]


def _plate(name: str):
    x, y, front = L.PLATES[name]
    shapes = [mk.cylinder((0.0, 0.0, L.PLATE_H / 2), L.PLATE_R, L.PLATE_H, color=L.PLATE_GREY),
              *seat_paint(front, "front"), *seat_paint(front, "rear"),
              *arrow(front, L.PLATE_ARROW, 0.0, L.PLATE_H), *arrow(front, L.BENCH_ARROW, L.BENCH_ARROW_DX, 0.0)]
    plate = mk.assembly(name, origin=(x, y, L.TOP), shapes=shapes, color=L.PLATE_GREY)  # no mass: fixed
    side = name.split("_")[0]
    anchor = Cylinder(f"{side}_anchor_post", pos=(x + L.ANCHOR_DX, y, L.TOP + L.PLATE_H + L.ANCHOR_H / 2),
                      radius=L.ANCHOR_R, height=L.ANCHOR_H, color=L.ANCHOR_STEEL)
    adjuster = Box(f"{side}_adjuster_block", pos=(x + L.ADJ_DX, y, L.TOP + L.PLATE_H + L.ADJ_SIZE[2] / 2),
                   size=L.ADJ_SIZE, color=L.ADJ_BRASS)
    return plate, anchor, adjuster


PLATES = [part for name in L.PLATES for part in _plate(name)]
OBJECTS = [*PLATES, *SHOES]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and a teleport back to the start are
    two more signals)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided. A trace
    that cannot be written is skipped (it must never break a run)."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """1.0 now: on both plates a primary shoe seated (fine: mid-arc point within 8 mm, yaw within 8 degrees, flat on
    the plate, let go and settled) on the seat toward the arrow and a secondary on the other seat, every shoe still
    over the last 1 s; held SUCCESS_HOLD_S = 1 s. This runs first every control step, so it also feeds the tracker
    (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update([objects[n] for n in SHOE_NAMES], arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    if isinstance(arms, dict):  # each arm on its own: a missing one is traced as None, never a crash
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """First match wins: 1.0 both plates correct (fine) with every shoe still for 1 s; 0.75 both plates correct
    (coarse: 14 mm, 15 degrees); 0.5 exactly one plate correct (coarse); 0.25 two or more shoes seated (coarse)
    anywhere; 0 otherwise. A plate is correct when its seat toward the arrow holds a primary and its other seat a
    secondary; a shoe counts only when let go and settled. Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in SHOE_NAMES], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.025  # 50 mm: with the TCP GRIP_INSET inside the mid-arc point, 12 mm clear of the shoe on each side
GRIP = -(2 * L.SHOE_HALF_W - 0.008) / 2  # 12 mm commanded across the shoe's 20 mm width
# Each shoe is gripped at its mid-arc block with the fingers closing across its width (radially). Gripper yaw 0
# closes along y, so the yaw is the shoe's minus 90 degrees (or plus 90). Every seat's radial direction is along y,
# so each shoe is laid at gripper yaw 0, and it is picked at the yaw that turns it the right way: minus the seat's
# yaw. A shoe for a +y seat is picked at yaw -90 deg, one for a -y seat at +90 (each within joint6's reach there:
# the right arm needs +90 at the inner pick and -90 at the outer, the left arm the mirror; the other sign at an
# inner pick runs joint6 into its +-120 deg limit).
# Trim of the place yaw, for the GPU run: in our GPU runs, parts carried in a gripper ended turned counter-clockwise
# from the commanded yaw by about +0.6 to +1.7 degrees, so every place yaw is set 1.15 degrees clockwise. Offline
# (no bias) each shoe is laid 1.15 degrees turned, inside the 8 degree window. If yaw_err_deg in TASK_TRACE after
# 'let go' is more than 2 degrees from 0, change YAW_TRIM by minus that many degrees.
YAW_TRIM = math.radians(-1.15)
GRIP_UP = 0.017  # the TCP 17 mm above the shoe's underside: the pad tips 4 mm above it (and the bench)
DROP = 0.001  # each shoe is let go with its underside 1 mm above its seat's paint
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.TOP + L.PLATE_H + L.PAINT_H + DROP + GRIP_UP
SLOW = 0.020  # the last 20 mm down onto a shoe, or onto a seat, are taken slowly
HOVER_Z = L.TOP + 0.100  # travel height: a carried shoe's underside 83 mm up, 69 mm over the anchor posts' tops
# Trim of the place points, for the GPU run: parts have landed about 4 mm short of their targets along the carry
# (arm lag under load). Each place point is moved 3 mm on along the carry's direction, so a landing 0-6 mm short
# ends within 3 mm of the seat. If off_seat_mm in TASK_TRACE is over 4 mm,
# move that arm's PLACE_TRIM by minus the offset.
PLACE_LEAD = 0.003
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
# The grip's inset, for the GPU run: closing across the shoe at mid-arc, the outer pad meets the flat 20 mm block
# face 10 mm out from the mid-arc point, but the inner pad (40 mm wide) meets the concave side first at its two edges,
# about 16 mm in (15.7 mm on the simulated shoe's discs). So the jaws centre on a point about 3 mm inside the mid-arc
# point, and a shoe gripped at its mid-arc point would be pushed that far outward along the bench by the inner pad
# first. The TCP therefore comes down GRIP_INSET inside the mid-arc point (both pads meet the shoe together; open,
# they are 12 mm from its outer face and 9 mm from its concave side), the shoe rides that far outward of the TCP in
# the hand, and the place point is set the same distance inside the seat's mid-arc point (toward the plate's
# centre). Offline the checker centres the grip block between the pads, so there each shoe lands GRIP_INSET inside
# its seat (with the lead, 1.3 to 2.2 mm off), inside the 8 mm window. In the simulator, with the shoe 3 mm outward
# in the hand, it should land about PLACE_LEAD on from its seat, less the arm's lag.
GRIP_INSET = 0.003
PARK = (0.200, 0.250, L.TOP + 0.150)  # each arm ends over its own side's emptied bench, clear of the plates
# Which arm takes which shoe to which seat, in order (both arms on one clock). The right arm starts with the inner
# shoe while the left arm takes its outer one, so the two grippers stay more than 0.30 m apart in y throughout.
PLAN = {
    "right": (("primary_1", ("right_plate", "front")), ("secondary_1", ("right_plate", "rear"))),
    "left": (("secondary_2", ("left_plate", "rear")), ("primary_2", ("left_plate", "front"))),
}


def place_point(name: str, seat, arm: str) -> tuple[float, float]:
    """The TCP's (x, y) over a seat: the seat's mid-arc point, led PLACE_LEAD on along the carry, GRIP_INSET in
    toward the plate's centre (the shoe rides that far outward of the TCP), plus the trim."""
    _, x0, y0, _ = L.SHOES[name]
    sx, sy, syaw = L.SEATS[seat]
    d = np.array([sx - x0, sy - y0])
    d = d / np.linalg.norm(d)
    ix, iy = GRIP_INSET * math.cos(math.radians(syaw)), GRIP_INSET * math.sin(math.radians(syaw))
    tx, ty = PLACE_TRIM[arm]
    return float(sx + PLACE_LEAD * d[0] - ix + tx), float(sy + PLACE_LEAD * d[1] - iy + ty)


def pick_point(name: str) -> tuple[float, float]:
    """The TCP's (x, y) at the pick: GRIP_INSET inside the shoe's mid-arc point, where the closing pads centre."""
    _, x, y, yaw = L.SHOES[name]
    return (float(x - GRIP_INSET * math.cos(math.radians(yaw))),
            float(y - GRIP_INSET * math.sin(math.radians(yaw))))


def pick_yaw(name: str, seat) -> float:
    """The gripper yaw (rad) at the pick: closing across the shoe's width, turning it onto the seat's yaw at
    gripper yaw 0."""
    return math.radians(L.SHOES[name][3] - L.SEATS[seat][2])


def _move(name: str, seat, arm: str, first: bool) -> list[Waypoint]:
    """Over a shoe, slowly down round it, grip it, lift it, carry it over its seat turning it, set it down slowly,
    let go, straight up."""
    x, y = pick_point(name)
    yp = pick_yaw(name, seat)
    yl = 0.0 + YAW_TRIM
    px, py = place_point(name, seat, arm)
    return [
        Waypoint(2.0 if first else 1.8, (x, y, HOVER_Z), OPEN, f"over {name}", yp),
        Waypoint(1.0, (x, y, PICK_Z + SLOW), OPEN, "down to just above it", yp),
        Waypoint(1.5, (x, y, PICK_Z), OPEN, "slowly down round it", yp),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", yp),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", yp),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", yp),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", yp),
        Waypoint(2.5, (px, py, HOVER_Z), GRIP, f"carry it over {seat[0]} {seat[1]} seat, turning it", yl),
        Waypoint(0.5, (px, py, HOVER_Z), GRIP, "settle over the seat", yl),
        Waypoint(1.0, (px, py, PLACE_Z + SLOW), GRIP, "down to just above the seat", yl),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly onto the seat", yl),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", yl),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", yl),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", yl),
        Waypoint(1.0, (px, py, HOVER_Z), OPEN, "straight up, clear of it", yl),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for k, (name, seat) in enumerate(PLAN[arm]):
        wps += _move(name, seat, arm, k == 0)
    sign = 1.0 if arm == "left" else -1.0
    return wps + [Waypoint(1.5, (PARK[0], sign * PARK[1], PARK[2]), OPEN, "park, clear of the plates", YAW_TRIM)]


def demo():
    """Both arms on one clock, each fitting its own side's plate. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
