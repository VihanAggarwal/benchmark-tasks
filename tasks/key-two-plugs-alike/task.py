"""On the station table: two grey aluminium training plugs (enlarged models of pin-tumbler lock plugs taken out of
their shells), fixed in front of each arm along x at y +24.0 and -24.0 cm, each 186 x 50 x 60 mm with four
open-topped 34 x 34 mm chambers in a row (chamber 1 nearest the robot, beside the key's bow). Each chamber's floor
is a light silver block standing for the new key's cut there; floors 20, 32, 14 and 26 mm high (code 3-1-4-2). On
the midline, a floorless grey pinning kit (5 rows x 2 columns of open cells) holds ten loose gold key pins, 24 mm
across and 28 to 52 mm long (size n in row n of each column; size 1 nearest). Task: key both plugs alike, one pin of
the code's size per chamber, every pin top flush with the plug top (the shear line), nothing else in or over a
chamber.

The scorer reads the arms (TCP) through rigstate.py, which wraps the embodiment's step and reset from this file's
import; it changes nothing the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker, tilt_deg, valid  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# The trial ends early once all eight chambers have been pinned for 1 s; otherwise the last step is scored.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Key both lock plugs alike to the code 3-1-4-2, chamber 1 being the one nearest the key's bow. Into each chamber "
    "drop one key pin of the size its code number gives, taken from the pin kit (size 1, the shortest, is in the "
    "kit's nearest row; size 5, the longest, is in its farthest), so that every pin stands on the key and ends flush "
    "with the top of its plug, with nothing else in or over the chamber."
)


def _box(name: str, x: float, y: float, size: tuple[float, float, float], color, z0: float = 0.0) -> Box:
    """A fixed box standing on the table (its bottom z0 above the table top)."""
    return Box(name, pos=(x, y, L.TOP + z0 + size[2] / 2), size=size, color=color)


def _plug(side: str) -> list[Box]:
    """One training plug: two side walls, five cross walls, four floor blocks (the key's cuts) and the key's bow."""
    s, y = side[0], L.PLUG_Y[side]
    parts = [
        _box(f"plug_{s}_side_{k}", L.PLUG_X, y + sign * L.SIDE_WALL_OFF, (L.PLUG_LEN, L.SIDE_WALL_T, L.PLUG_H),
             L.PLUG_GREY)
        for k, sign in (("in", -1.0 if side == "left" else 1.0), ("out", 1.0 if side == "left" else -1.0))
    ]
    parts += [_box(f"plug_{s}_wall_{j}", x, y, (L.CROSS_WALL_T, L.CHAMBER, L.PLUG_H), L.PLUG_GREY)
              for j, x in enumerate(L.CROSS_WALL_X)]
    parts += [_box(f"key_{s}_cut_{i + 1}", x, y, (L.CHAMBER, L.CHAMBER, h), L.KEY_SILVER)
              for i, (x, h) in enumerate(zip(L.CHAMBER_X, L.FLOOR_H))]
    parts.append(_box(f"key_{s}_bow", L.BOW_X, y, L.BOW_SIZE, L.KEY_SILVER))
    return parts


def _kit() -> list[Box]:
    """The pinning kit: three full-length walls along x and twelve 56 mm segments across y."""
    long = (L.KIT_X1 - L.KIT_X0, L.KIT_WALL_T, L.KIT_WALL_H)
    cross = (L.KIT_WALL_T, L.KIT_CELL[1], L.KIT_WALL_H)
    parts = [_box(f"kit_long_{j}", (L.KIT_X0 + L.KIT_X1) / 2, y, long, L.KIT_GREY)
             for j, y in enumerate(L.KIT_LONG_Y)]
    parts += [_box(f"kit_cross_{c[0]}{j}", x, L.KIT_COL_Y[c], cross, L.KIT_GREY)
              for c in L.SIDES for j, x in enumerate(L.KIT_CROSS_X)]
    return parts


PINS = [
    Cylinder(name, pos=(*L.PIN_START[name], L.TOP + L.PIN_LEN[n] / 2), radius=L.PIN_R, height=L.PIN_LEN[n],
             mass=L.PIN_MASS[n], color=L.PIN_GOLD)
    for name, (_, n) in L.PINS.items()
]
OBJECTS = [*_plug("left"), *_plug("right"), *_kit(), *PINS]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
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


def _pins(objects) -> dict:
    return {name: objects.get(name) for name in L.PINS}


def success(objects) -> bool:
    """All eight chambers pinned: in each, one pin standing within 8 degrees of vertical with its top-face centre
    within 2.5 mm of the plug top, still (path under 2 mm in 0.5 s), both TCPs at least 50 mm from that top, and no
    other pin over the chamber; held 1 s. This runs first every control step, so it also feeds the tracker (with
    the arms rigstate read for this step)."""
    pins = _pins(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(pins, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_opening=arms["left"].opening, right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    row["pin_pos"] = {n: (s.pos if s is not None else None) for n, s in pins.items()}
    # every pin's lean, wherever it is (a pin caught on a wall top is in no chamber's row)
    row["pin_tilt_deg"] = {n: (round(tilt_deg(s), 2) if valid(s) else None) for n, s in pins.items()}
    _trace(row)
    return result.success


def score(objects) -> float:
    """k / 8, k the chambers pinned over both plugs (0, 0.125, ..., 1.0)."""
    return _TRACKER.result(_pins(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap. 'Open' is a 40 mm gap everywhere (the gripper
# is never opened fully near the kit); 'closed' is commanded to 16 mm, 8 mm narrower than the 24 mm pin.
OPEN = -0.020
CLOSED = -0.008
YAW = 0.0  # the fingers close along y
CARRY_Z = L.TOP + 0.130  # travel and carry: a carried pin's bottom clears the plug top by 23 mm or more
GRIP_ABOVE = 0.001  # at the kit the TCP stops 1 mm above the pin's top: the pads hold its top 12 mm
PLACE_Z = L.TOP + 0.079  # release: the pin's top at 78 mm, its bottom 10-28 mm inside the chamber with 5 mm
#                          clearance a side, the pad tips 6 mm above the plug top; it drops 18 mm onto its floor
FIRST_TRAVEL_S = 3.0  # the first move, from the ready pose
RIGHT_START_S = 7.5  # the right arm starts its first pin this much later than the left
RIGHT_WAIT = (L.CHAMBER_X[0], L.PLUG_Y["right"], CARRY_Z)  # it waits over its own plug (the ready pose is not
#                                                         reachable with a straight-down grip)


def _one_pin(side: str, i: int, first: bool) -> list[Waypoint]:
    """Take the code's pin for chamber i from this side's kit column and drop it into the chamber (14.5 s)."""
    size = L.CODE[i]
    kx, ky = L.KIT_ROW_X[size - 1], L.KIT_COL_Y[side]
    cx, cy = L.CHAMBER_X[i], L.PLUG_Y[side]
    grip_z = L.TOP + L.PIN_LEN[size] + GRIP_ABOVE
    return [
        Waypoint(FIRST_TRAVEL_S if first else 2.0, (kx, ky, CARRY_Z), OPEN, f"over kit row {size}", YAW),
        Waypoint(1.5, (kx, ky, grip_z), OPEN, f"down onto the size-{size} pin", YAW),
        Waypoint(0.5, (kx, ky, grip_z), OPEN, "settle", YAW),
        Waypoint(1.5, (kx, ky, grip_z), CLOSED, "grip its top 12 mm", YAW),
        Waypoint(1.5, (kx, ky, CARRY_Z), CLOSED, "lift it out of the kit", YAW),
        Waypoint(2.5, (cx, cy, CARRY_Z), CLOSED, f"carry it over chamber {i + 1}", YAW),
        Waypoint(0.5, (cx, cy, CARRY_Z), CLOSED, "settle over the chamber", YAW),
        Waypoint(1.5, (cx, cy, PLACE_Z), CLOSED, "lower it into the chamber", YAW),
        Waypoint(0.5, (cx, cy, PLACE_Z), CLOSED, "settle", YAW),
        Waypoint(1.5, (cx, cy, PLACE_Z), OPEN, "let go: it drops 18 mm onto the key's cut", YAW),
        Waypoint(1.0, (cx, cy, CARRY_Z), OPEN, "straight up", YAW),
    ]


def _arm(side: str) -> list[Waypoint]:
    """Chambers 1-4 in turn: kit rows 3, 1, 4, 2."""
    out = []
    for i in range(4):
        out += _one_pin(side, i, first=(i == 0))
    return out


def _left() -> list[Waypoint]:
    return _arm("left")


def _right() -> list[Waypoint]:
    """The left arm's mirror image, 7.5 s behind it: the arms take turns at the shared kit."""
    wait = [
        Waypoint(3.0, RIGHT_WAIT, OPEN, "over its plug", YAW),
        Waypoint(RIGHT_START_S - 3.0, RIGHT_WAIT, OPEN, "wait while the left arm is at the kit", YAW),
    ]
    return wait + _arm("right")


def demo():
    """Both arms on one clock: each keys the plug in front of it, taking turns at the kit. DEMO_HOLD_ONLY=1: no
    plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped
    (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
