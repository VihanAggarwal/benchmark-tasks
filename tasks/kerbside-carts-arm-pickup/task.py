"""On the station table, at 1:10 scale: a residential street. The table top is the verge; across the far side runs a
raised kerb (face at x 0.390, 12 mm tall) with the road beyond it. On the verge stand a mailbox on its post (on the
midline) and a fire hydrant (to the left); a car is parked in the road at the kerb to the right; on the far left the
house's driveway crosses the verge to the road. Two wheeled refuse carts (dark grey bodies; green lid on the left,
blue lid on the right; 79 x 58 mm footprint, 101 mm tall, 120 g, loose) stand near the house, side-on, their fronts
toward each other. Rule (automated side-loader collection): each cart at the kerb, its front 0-30 mm behind the kerb
face and not on it, lid opening toward the street (within 25 degrees), off the driveway, and 91 mm (3 ft at scale)
clear of the other cart, the mailbox and its post, the hydrant and the parked car.

Each cart is a mechkit compound (mechkit.py, next to this file: one loose rigid body of nine boxes: body, lid, handle
bar, two wheels and a white lid arrow), so its front can be seen and the scorer can read which way it faces. It
changes nothing else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the carts are built as compound bodies)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, both carts standing still at the kerb) must hold 3 s, not the default 1 s. A cart counts as standing
# once its lowest point is within 3 mm of the verge, which happens at the end of the slow descent while the grippers
# still hold the carts (sooner on the GPU, where the arms sag a few mm under the carts); settling, letting go and
# rising take 0.6 + 1.5 + 1.5 s, so the trial ends with the fingers open, going up.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Set both wheeled carts out for the automated refuse truck. Stand each cart on the verge with its lid opening "
    "toward the street (the white arrow, square to the kerb within 25 degrees; handle and wheels toward the house), "
    "its front 0 to 30 mm (0 to 0.3 m at this 1:10 scale) behind the kerb and not on it. Keep each cart at least "
    "91 mm (3 feet) from the other cart, the mailbox and its post, the hydrant and the parked car, and off the "
    "driveway."
)


def _span_box(name, xs, ys, zs, colour) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    (x0, x1), (y0, y1), (z0, z1) = xs, ys, zs
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=L.COLOURS[colour])


KERB = _span_box("kerb", L.KERB_X, L.KERB_Y, (0.0, L.KERB_H), "kerb")
ROAD = _span_box("road", L.ROAD_X, L.ROAD_Y, (0.0, L.ROAD_H), "road")
DRIVEWAY = _span_box("driveway", L.DRIVE_X, L.DRIVE_Y, (0.0, L.DRIVE_H), "drive")
POST = Cylinder("mailbox_post", pos=(L.POST_XY[0], L.POST_XY[1], L.TOP + L.POST_H / 2), radius=L.POST_R,
                height=L.POST_H, color=L.COLOURS["post"])
MAILBOX = _span_box("mailbox", L.MAILBOX[0:2], L.MAILBOX[2:4], L.MAILBOX[4:6], "mailbox")
HYDRANT = Cylinder("hydrant", pos=(L.HYDRANT_XY[0], L.HYDRANT_XY[1], L.TOP + L.HYDRANT_H / 2), radius=L.HYDRANT_R,
                   height=L.HYDRANT_H, color=L.COLOURS["hydrant"])
HYDRANT_BONNET = Cylinder("hydrant_bonnet", pos=(L.HYDRANT_XY[0], L.HYDRANT_XY[1],
                                                 L.TOP + L.HYDRANT_H + L.HYDRANT_CAP_H / 2),
                          radius=L.HYDRANT_CAP_R, height=L.HYDRANT_CAP_H, color=L.COLOURS["hydrant"])
CAR = _span_box("car", L.CAR_BODY[0:2], L.CAR_BODY[2:4], L.CAR_BODY[4:6], "car")
CAR_CABIN = _span_box("car_cabin", L.CAR_CABIN[0:2], L.CAR_CABIN[2:4], L.CAR_CABIN[4:6], "car_glass")
STREET = [KERB, ROAD, DRIVEWAY, POST, MAILBOX, HYDRANT, HYDRANT_BONNET, CAR, CAR_CABIN]


def _cart(name: str):
    x, y, heading = L.START[name]
    lid = L.LID_COLOUR[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP),
        yaw=heading,
        mass=L.CART_MASS,
        color=L.COLOURS["body"],
        shapes=[mk.box(c, s, color=L.COLOURS[lid if k == "lid" else k]) for _, c, s, k in L.CART_SHAPES],
    )


CART_LEFT = _cart("cart_left")
CART_RIGHT = _cart("cart_right")
OBJECTS = [*STREET, CART_LEFT, CART_RIGHT]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


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
    """Both carts set out (score 1.0): each standing still at the kerb, facing the street, off the driveway and 91 mm
    clear of the other cart and the street furniture; held 3 s. This runs first every control step, so it also feeds
    the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    left, right = objects.get("cart_left"), objects.get("cart_right")
    result = _TRACKER.update(left, right)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(left, "source", None),
                   origins={"cart_left": left.origin, "cart_right": right.origin},
                   quats={"cart_left": left.quat_xyzw, "cart_right": right.quat_xyzw},
                   low_z={"cart_left": left.low[2], "cart_right": right.low[2]},
                   vels={"cart_left": left.lin_vel, "cart_right": right.lin_vel})
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if either cart is knocked over (tilted over 45 degrees) or any part of a cart is more than 2 mm past the kerb
    face (on the kerb or in the road). Otherwise, per cart: 0.25 if it stands (tilted 10 degrees or less, lowest point
    within 3 mm of the verge, still over 0.5 s) off the driveway, its front within 25 degrees of the street direction
    and its most forward point 0-30 mm behind the kerb face; 0.5 if it is also 91 mm or more from the other cart, the
    mailbox and its post, the hydrant and the car (top view); else 0. Read from the state at hand (the last step)."""
    return _TRACKER.result(objects.get("cart_left"), objects.get("cart_right")).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Both arms at once, each with its own cart: the left arm sets the green-lidded cart between the mailbox and the
# hydrant, the right arm the blue-lidded one between the mailbox and the car. Each grip is across the cart's 58 mm
# body at its middle (the frame origin, 0.7 mm in front of the centre of mass), and the quarter turn is made with the
# wrist while the cart is carried: the cart's heading changes with the grip's yaw. Yaw 0 closes the fingers along
# world y, so a cart facing +x is held at yaw 0 and one facing -y (+y) at yaw -90 (+90) degrees; near its own side
# the left arm uses negative yaws and the right arm positive ones.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # 94 mm: 18 mm clear of the body on each side
GRIP = -(2 * L.BODY_V - 0.008) / 2  # 50 mm commanded on the 58 mm body
GRIP_U = 0.0  # the grip's centre, forward of the cart's origin (m, cart frame)
# The TCP 78 mm up: the pads (from 25 mm above the TCP to their tips 13 mm below it) on the body and lid from 65 mm to
# the lid's top (101 mm), and the gripper's body (from 35 mm above the TCP) 11 mm over the lid arrow (102 mm).
GRIP_Z = L.TOP + 0.078
CARRY_Z = GRIP_Z + 0.025  # the cart lifted 25 mm
PLACE_Z = GRIP_Z + 0.002  # let go with the cart's lowest point 2 mm above the verge
APPROACH_Z = GRIP_Z + 0.030  # the last 30 mm down to the grip are slow
ABOVE_Z = L.TOP + 0.140  # pad tips 127 mm up, 25 mm over the lid arrow
PARK = {"left": (0.200, 0.300, L.TOP + 0.150), "right": (0.200, -0.300, L.TOP + 0.150)}
SIDE = {"left": "cart_left", "right": "cart_right"}


def grip_point(origin_xy, heading: float) -> tuple[float, float]:
    """Where the grip is (world x, y) for a cart whose origin is at ``origin_xy`` facing ``heading``."""
    return (origin_xy[0] + GRIP_U * math.cos(heading), origin_xy[1] + GRIP_U * math.sin(heading))


def grip_yaw(heading: float) -> float:
    """The grip's yaw for a cart facing ``heading``: the fingers close across the cart (along its left-right)."""
    return math.remainder(heading, 2 * math.pi)


def _arm(arm: str) -> list[Waypoint]:
    """One arm: over its cart, down slowly, grip across the body, lift 25 mm, carry it to the kerb turning it a
    quarter turn to face the street, lower it slowly, let go, straight up, park."""
    cart = SIDE[arm]
    x0, y0, h0 = L.START[cart]
    x1, y1, h1 = L.TARGET[cart]
    sx, sy = grip_point((x0, y0), h0)
    px, py = grip_point((x1, y1), h1)
    ys, yp = grip_yaw(h0), grip_yaw(h1)
    return [
        Waypoint(3.0, (sx, sy, ABOVE_Z), OPEN, f"over {cart}", ys),
        Waypoint(1.5, (sx, sy, APPROACH_Z), OPEN, "down beside its body", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), OPEN, "down the last 30 mm slowly", ys),
        Waypoint(0.5, (sx, sy, GRIP_Z), OPEN, "settle", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, "grip across the body", ys),
        Waypoint(1.5, (sx, sy, CARRY_Z), GRIP, "lift 25 mm", ys),
        Waypoint(3.5, (px, py, CARRY_Z), GRIP, "carry it to the kerb, turning its lid toward the street", yp),
        Waypoint(0.5, (px, py, CARRY_Z), GRIP, "settle over the place", yp),
        Waypoint(2.0, (px, py, PLACE_Z), GRIP, "lower it slowly", yp),
        Waypoint(0.6, (px, py, PLACE_Z), GRIP, "settle", yp),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go", yp),
        Waypoint(1.5, (px, py, ABOVE_Z), OPEN, "straight up", yp),
        Waypoint(2.0, PARK[arm], OPEN, "out of the way"),
    ]


def demo():
    """Both arms on one clock, each setting out its own cart. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
