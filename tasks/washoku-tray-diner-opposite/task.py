"""On the station table: a dark red lacquer meal tray (220 x 300 mm) on the midline, and beyond its far edge an indigo
cushion where the guest sits, facing the robot. On a light wood serving board at each side wait the four dishes of a
Japanese set meal (teishoku): on the right (-y) a cream-white rice bowl and a small blue side-dish bowl; on the left
(+y) a black lacquer soup bowl with a red rim and a white rectangular plate with a grilled fish on it, the fish's head
pointing away from the robot. Rule (the basic washoku layout, laid for the guest): seen from the guest, rice front
left, soup front right, the fish plate back right, the side dish back left, and the fish lying with its head to the
guest's left and its belly toward the guest. The guest faces -x, so the guest's left is -y and the guest's front is
the tray's far half. Each dish placed and let go earns 0.25.

Every dish is a loose mechkit compound (mechkit.py, next to this file): one rigid body, so a bowl can carry a disc of
food on top and the fish plate its fish, head and belly marks, and the scorer can read each dish's pose (base frame
and orientation). The scorer also reads the arms (pad-tip points, openings) through rigstate.py, which wraps the
embodiment's step and reset from this file's import, only to check that each dish was let go; neither changes
anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the dishes are built as compound bodies)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0: every dish placed, let go and still for 1 s) must hold this long. Should the arms ever be unreadable,
# the last dish set down could count from the end of its slow descent while still gripped; settling and letting go
# take 0.5 + 1.5 s, so with 2.5 s the trial still cannot end while the fingers are closed on a dish.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Set this teishoku tray for the guest who sits on the indigo cushion across the table, facing you. The white "
    "bowl is rice, the black lacquer bowl is miso soup, the small blue bowl is the side dish, and the plate holds a "
    "whole grilled fish. Put all four on the tray the way a Japanese set meal is laid for the guest, then let go."
)

OBJECT_OF = {"rice": "rice_bowl", "soup": "soup_bowl", "side": "side_dish_bowl", "fish": "fish_plate"}


def _span_box(name, x0, x1, y0, y1, h, color) -> Box:
    """A fixed flat box given by its extents (x, y in the world), standing on the table, h tall."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + h / 2), size=(x1 - x0, y1 - y0, h), color=color)


TRAY = _span_box("meal_tray", L.TRAY_X0, L.TRAY_X1, L.TRAY_Y0, L.TRAY_Y1, L.TRAY_H, L.TRAY_RED)
CUSHION = _span_box("guest_cushion", L.CUSHION_X0, L.CUSHION_X1, L.CUSHION_Y0, L.CUSHION_Y1, L.CUSHION_H,
                    L.CUSHION_INDIGO)
BOARDS = [
    _span_box(f"serving_board_{side}", L.BOARD_X0, L.BOARD_X1, *sorted((s * L.BOARD_Y_IN, s * L.BOARD_Y_OUT)),
              L.BOARD_H, L.BOARD_WOOD)
    for side, s in (("left", 1.0), ("right", -1.0))
]


def _origin(name: str) -> tuple[float, float, float]:
    x, y, _ = L.START[name]
    return (x, y, L.TOP + L.BOARD_TOP)


def _bowl(name: str, radius: float, body_h: float, body_color, food_r: float, food_color, mass: float,
          rim_h: float = 0.0, rim_color=None):
    """A bowl: a cylinder (its body), an optional band of another colour on top (a lacquered rim), and a thin disc of
    food 1 mm proud of the rim, as one loose rigid body standing on its serving board."""
    shapes = [mk.cylinder((0.0, 0.0, body_h / 2), radius, body_h, color=body_color)]
    if rim_h:
        shapes.append(mk.cylinder((0.0, 0.0, body_h + rim_h / 2), radius, rim_h, color=rim_color))
    shapes.append(mk.cylinder((0.0, 0.0, body_h + rim_h + L.FOOD_H / 2), food_r, L.FOOD_H, color=food_color))
    return mk.assembly(OBJECT_OF[name], origin=_origin(name), yaw=L.START[name][2], mass=mass, color=body_color,
                       shapes=shapes)


RICE = _bowl("rice", L.RICE_R, L.RICE_BODY_H, L.PORCELAIN, L.RICE_FOOD_R, L.RICE_WHITE, L.RICE_MASS)
SOUP = _bowl("soup", L.SOUP_R, L.SOUP_BODY_H, L.LACQUER_BLACK, L.SOUP_FOOD_R, L.MISO_BROWN, L.SOUP_MASS,
             rim_h=L.SOUP_RIM_H, rim_color=L.LACQUER_RED)
SIDE = _bowl("side", L.SIDE_R, L.SIDE_BODY_H, L.SIDE_BLUE, L.SIDE_FOOD_R, L.GREENS, L.SIDE_MASS)
FISH = mk.assembly(
    OBJECT_OF["fish"],
    origin=_origin("fish"),
    yaw=L.START["fish"][2],
    mass=L.FISH_MASS,
    color=L.PLATE_WHITE,
    shapes=[
        mk.box((0.0, 0.0, L.PLATE_T / 2), (L.PLATE_L, L.PLATE_W, L.PLATE_T), color=L.PLATE_WHITE),
        mk.box(L.FISH_BODY_CENTER, L.FISH_BODY_SIZE, color=L.FISH_GRILLED),
        mk.box(L.FISH_HEAD_CENTER, L.FISH_HEAD_SIZE, color=L.FISH_DARK),
        mk.box(L.FISH_TAIL_CENTER, L.FISH_TAIL_SIZE, color=L.FISH_DARK),
        mk.box(L.FISH_BELLY_CENTER, L.FISH_BELLY_SIZE, color=L.FISH_BELLY),
        mk.box(L.FISH_RIDGE_CENTER, L.FISH_RIDGE_SIZE, color=L.FISH_DARK),
    ],
)
DISH_SPECS = {"rice": RICE, "soup": SOUP, "side": SIDE, "fish": FISH}
OBJECTS = [TRAY, CUSHION, *BOARDS, RICE, SIDE, SOUP, FISH]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and every dish teleported back to its
    start are two more signals)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided. A trace that
    cannot be written is skipped (it must never break a run)."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def _dishes(objects) -> dict:
    return {n: objects.get(OBJECT_OF[n]) for n in L.DISHES}


def success(objects) -> bool:
    """1.0 now: every dish in its quadrant seen from the guest (rice front left, soup front right, fish back right,
    side dish back left), on the tray, upright, let go, still for 1 s, none overlapping, none ever tipped, the fish
    head to the guest's left and belly toward the guest; held SUCCESS_HOLD_S = 2.5 s. This runs first every control
    step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(_dishes(objects), arms)
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
    """0.25 for each dish placed: its centre at least 10 mm inside its quadrant seen from the guest, its footprint on
    the tray (5 mm overhang allowed), standing on the tray top (lowest point within 3 mm), upright (10 degrees), let
    go (no gripper's pad-tip point lower than 10 mm above its top within finger reach), still (centre path under
    3 mm over the last 1 s); the fish plate also head within 15 degrees of the guest's left and belly within 15
    degrees of the guest. A dish overlapping another (both 0) or that ever tipped past 45 degrees earns 0. Read from
    the state at hand (the last step)."""
    return _TRACKER.result(_dishes(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # fully open, 94 mm: 14 to 19 mm either side of a bowl, 17 mm either side of the plate
GRIP = {
    "rice": -(2 * L.RICE_R - 0.008) / 2,  # 58 mm across the 66 mm bowl
    "soup": -(2 * L.SOUP_R - 0.008) / 2,  # 56 mm across 64
    "side": -(2 * L.SIDE_R - 0.008) / 2,  # 47 mm across 55
    "fish": -(L.PLATE_W - 0.008) / 2,  # 52 mm across the plate's 60 mm width, at its middle, under the fish
}
# The TCP this far above the dish's bottom (the pad tips 13 mm lower): the bowls are held on their lower part; the
# plate by its two long edges, the pad tips 3 mm above its underside, the pads 12 mm clear of the fish either side.
GRIP_UP = {"rice": 0.025, "soup": 0.027, "side": 0.020, "fish": 0.016}
DROP = 0.001  # each dish is let go with its bottom 1 mm above the tray
SLOW = 0.020  # the last 20 mm down onto a dish, or onto the tray, are taken slowly
HOVER_Z = L.TOP + 0.095  # travel height: a carried dish's bottom 68 to 79 mm up, 12 mm or more over any dish's top
UP_Z = L.TOP + 0.095  # straight up after letting go: pad tips 31 mm or more over the dish's top
PARK = (0.200, 0.250, L.TOP + 0.150)  # each arm ends on its own side, clear of the tray (y mirrored)
# Where each dish goes (the place's centre): 55 mm inside the centre line across, 75 mm from the one front to back.
TARGET = {"rice": (0.385, -0.075), "soup": (0.385, 0.075), "side": (0.275, -0.075), "fish": (0.275, 0.075)}
# Gripper yaw (yaw 0 closes the fingers along y). The fish plate is gripped across its width at its start yaw (0) and
# turned to yaw -90 deg while carried, so the fish's head swings from +x to -y (the guest's left) and its belly from
# +y to +x (toward the guest); 1 degree clockwise of that, for the +0.6..+1.7 degree turn parts take in the hand.
YAW_TRIM = math.radians(-1.0)
PICK_YAW = {"rice": 0.0, "soup": 0.0, "side": 0.0, "fish": L.START["fish"][2]}
PLACE_YAW = {"rice": 0.0, "soup": 0.0, "side": 0.0, "fish": -math.pi / 2 + YAW_TRIM}
# Each place point leads its target by 4 mm along the way the dish travels (parts carried in a gripper landed about
# 4 mm short in our GPU runs). PLACE_TRIM moves an arm's place points if the GPU trace shows them off by a steady
# amount (read each dish's centre_mm in TASK_TRACE and move by minus the offset).
LEAD = 0.004
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
# Each arm lays its own side's two dishes, both arms on one clock: the right arm the rice (front left) and then the
# side dish (back left), the left arm the fish (back right) and then the soup (front right). Only one arm works over
# the tray at a time (two wrists over the tray at once come within 1 cm of each other): before picking a dish an arm
# waits WAIT s over it, so the arms take turns at the tray: rice, fish, side dish, soup.
PLAN = {"right": ("rice", "side"), "left": ("fish", "soup")}
WAIT = {("right", "rice"): 0.0, ("left", "fish"): 10.6, ("right", "side"): 5.1, ("left", "soup"): 4.1}


def place_point(name: str, arm: str) -> tuple[float, float]:
    """The commanded place point: the target, led 4 mm along the dish's way from its start, plus the arm's trim."""
    (tx, ty), (sx, sy, _) = TARGET[name], L.START[name]
    d = math.hypot(tx - sx, ty - sy)
    ox, oy = PLACE_TRIM[arm]
    return tx + LEAD * (tx - sx) / d + ox, ty + LEAD * (ty - sy) / d + oy


def _move(name: str, arm: str, first: bool) -> list[Waypoint]:
    """Over a dish (waiting there for the other arm if WAIT says so), slowly down round it, grip it, lift it, carry it over its place (the fish plate turning), settle,
    set it down slowly on the tray, let go, straight up."""
    x, y, _ = L.START[name]
    px, py = place_point(name, arm)
    pick_z = L.TOP + L.BOARD_TOP + GRIP_UP[name]
    place_z = L.TOP + L.TRAY_TOP + DROP + GRIP_UP[name]
    a, b, g = PICK_YAW[name], PLACE_YAW[name], GRIP[name]
    return [
        Waypoint(2.5 if first else 2.0, (x, y, HOVER_Z), OPEN, f"over the {name}", a),
        *([Waypoint(WAIT[(arm, name)], (x, y, HOVER_Z), OPEN, "wait for the other arm to clear the tray", a)]
          if WAIT[(arm, name)] > 0 else []),
        Waypoint(1.0, (x, y, pick_z + SLOW), OPEN, "down to just above it", a),
        Waypoint(1.5, (x, y, pick_z), OPEN, "slowly down round it", a),
        Waypoint(0.5, (x, y, pick_z), OPEN, "settle", a),
        Waypoint(1.5, (x, y, pick_z), g, f"grip the {name}", a),
        Waypoint(0.3, (x, y, pick_z), g, "settle the grip", a),
        Waypoint(1.2, (x, y, HOVER_Z), g, "lift it straight up", a),
        Waypoint(2.5 if name == "fish" else 2.0, (px, py, HOVER_Z), g,
                 "carry it over its place, turning it" if a != b else "carry it over its place", b),
        Waypoint(0.8, (px, py, HOVER_Z), g, "settle over its place", b),
        Waypoint(1.0, (px, py, place_z + SLOW), g, "down to just above the tray", b),
        Waypoint(1.5, (px, py, place_z), g, "slowly set it on the tray", b),
        Waypoint(0.5, (px, py, place_z), g, "settle", b),
        Waypoint(1.5, (px, py, place_z), OPEN, "let go of it", b),
        Waypoint(0.3, (px, py, place_z), OPEN, "settle", b),
        Waypoint(1.0, (px, py, UP_Z), OPEN, "straight up, clear of it", b),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for k, name in enumerate(PLAN[arm]):
        wps += _move(name, arm, k == 0)
    sign = 1.0 if arm == "left" else -1.0
    return wps + [Waypoint(1.5, (PARK[0], sign * PARK[1], PARK[2]), OPEN, "park, clear of the tray",
                           PLACE_YAW[PLAN[arm][-1]])]


def demo():
    """Both arms on one clock, each laying its own side's dishes. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
