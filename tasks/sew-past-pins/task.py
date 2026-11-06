"""On the station table, 40 cm in front of the arm bases and running left to right: a grey machine bed (120 mm wide,
970 mm long, fixed) with a cream frame plate (70 mm wide, 760 mm long, 10 mm thick) on a rail along its top: two plies
of a seam held in a sliding frame, as in free-motion quilting or darning with the feed dogs dropped, where the
operator moves the work under the needle. A blue seam line runs down its middle. A black handle (30 x 30 x 40 mm)
stands on its left end, just left of the left arm; three red pins (18 mm across, 40 mm tall, loose) stand in pockets
on the seam line near its right end. The plate slides toward the right, 18 cm at most, and only when it is pushed or
pulled. Over the bed on the right stands a fixed needle head: two posts and a cross bar, with a steel needle hanging
from the bar down to 8 mm above the plate. On the bed beside the plate a red mark shows the needle line and a yellow
band the pull window, 3 to 7 cm before it. A black pin dish stands on the table in front of the right arm, nearer
the arms than the bed.

Sew the seam: move the plate 16 cm or more toward the needle by its handle, gripping each pin and pulling it out of
the plate while it is beside the yellow band and putting it in the dish; no pin may reach the needle, be carried
under the needle head or be knocked loose, and the plate may move only while a gripper holds its handle. Only the
left arm reaches the handle at the start; only the right arm reaches the window and the dish.

The work (bed and plate) is a mechkit assembly (mechkit.py, next to this file: a fixed-base articulation with one
prismatic joint); the scorer reads the feed from it, and the arms (tool point, finger tips, opening) through
rigstate.py, which wraps the embodiment's step and reset from this file's import. Neither changes what the
embodiment does. offline_model.py is for our offline checker only (see _offline_checker_states below).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the work is built as an articulation)
import numpy as np  # noqa: E402
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with the seam complete, all three pins in the dish and both grippers let go (open, or clear of the
# handle and the pins): the trial cannot end while a gripper still holds the handle or a pin.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Sew the seam: move the cream plate at least 16 cm toward the needle by its black handle only. Grip each red pin "
    "and pull it out of the plate while it is beside the yellow bands before the needle, not earlier and not later, "
    "and put it in the black dish; never knock a pin loose. No pin may reach the needle or be carried under the "
    "needle head, and the plate may move only while a gripper holds its handle. Finish with both grippers open or "
    "clear of the handle and the pins."
)

# -- the work: one fixed assembly; shapes in its base frame (origin on the seam line at y 0, on the table top) --------
ORIGIN = (L.X_SEAM, 0.0, L.TOP)


def _local(x: float, y: float, z: float) -> tuple[float, float, float]:
    """A world (x, y) and a height above the table, in the work's base frame."""
    return (x - ORIGIN[0], y - ORIGIN[1], z)


def _marks() -> list:
    """Marks on the bed's top beside the plate, on both of its strips (20 mm wide, 1 mm high): a red mark on the
    needle line, and a yellow band over the pull window, from 30 to 70 mm before it."""
    w0, w1 = L.WINDOW
    out = []
    for side in (-1, 1):
        x = L.X_SEAM + side * L.MARK_DX
        out.append(mk.box(_local(x, L.Y_NEEDLE, L.BED_TOP + L.MARK_T / 2), (L.MARK_W, L.NEEDLE_MARK, L.MARK_T),
                          color=L.MARK_RED))
        out.append(mk.box(_local(x, L.Y_NEEDLE + (w0 + w1) / 2, L.BED_TOP + L.MARK_T / 2),
                          (L.MARK_W, w1 - w0, L.MARK_T), color=L.MARK_YELLOW))
    return out


BED = [mk.box(_local(L.X_SEAM, (L.BED_Y0 + L.BED_Y1) / 2, L.BED_T / 2), (L.BED_W, L.BED_Y1 - L.BED_Y0, L.BED_T)),
       *_marks()]


def _plate() -> list:
    """The frame plate as drawn at feed 0: a 4 mm bottom layer, a 6 mm top layer with three 28 mm square pockets on
    the seam line, the blue seam line on top (between the pockets and up to the handle), and the handle."""
    y0, y1 = L.PLATE_Y0, L.PLATE_Y1
    bottom_t = L.POCKET_FLOOR - L.BED_TOP
    top_t = L.PLATE_TOP - L.POCKET_FLOOR
    z_top = L.POCKET_FLOOR + top_t / 2
    side_w = (L.PLATE_W - L.POCKET) / 2
    shapes = [mk.box(_local(L.X_SEAM, (y0 + y1) / 2, L.BED_TOP + bottom_t / 2), (L.PLATE_W, y1 - y0, bottom_t))]
    for side in (-1, 1):  # the strips beside the pockets, the plate's whole length
        shapes.append(mk.box(_local(L.X_SEAM + side * (L.POCKET + side_w) / 2, (y0 + y1) / 2, z_top),
                             (side_w, y1 - y0, top_t)))
    edges = [y0]
    for d in L.PIN_D0.values():
        yc = L.Y_NEEDLE + d
        edges += [yc - L.POCKET / 2, yc + L.POCKET / 2]
    edges.append(y1)
    handle_lo = L.HANDLE_Y - L.HANDLE[1] / 2
    for a, b in zip(edges[0::2], edges[1::2]):  # the middle strip between the pockets
        shapes.append(mk.box(_local(L.X_SEAM, (a + b) / 2, z_top), (L.POCKET, b - a, top_t)))
        end = min(b, handle_lo)
        if end > a:  # the seam line on it, stopping at the handle
            shapes.append(mk.box(_local(L.X_SEAM, (a + end) / 2, L.PLATE_TOP + L.SEAM_LINE_T / 2),
                                 (L.SEAM_LINE_W, end - a, L.SEAM_LINE_T), color=L.SEAM_BLUE))
    shapes.append(mk.box(_local(L.X_SEAM, L.HANDLE_Y, L.HANDLE_Z), L.HANDLE, color=L.HANDLE_BLACK))
    return shapes


FEED = mk.slider("feed", "-Y", _plate(), upper=L.FEED_MAX, mass=L.PLATE_MASS, damping=L.FEED_DAMPING,
                 color=L.PLATE_CREAM)
WORK = mk.assembly("work", origin=ORIGIN, shapes=BED, joints=[FEED], color=L.BED_GREY)

PIN_OBJECTS = [Cylinder(n, pos=L.PIN_START[n], radius=L.PIN_R, height=L.PIN_H, mass=L.PIN_MASS, color=L.PIN_RED)
               for n in L.PINS]

_POST_H = L.BAR_Z0
NEEDLE_HEAD = [
    *[Box(f"needle_post_{name}", pos=(L.X_SEAM + side * L.POST_DX, L.HEAD_Y, L.TOP + _POST_H / 2),
          size=(L.POST[0], L.POST[1], _POST_H), color=L.HEAD_GREY) for name, side in (("near", -1), ("far", 1))],
    Box("needle_bar", pos=(L.X_SEAM, L.HEAD_Y, L.TOP + (L.BAR_Z0 + L.BAR_Z1) / 2),
        size=(L.BAR_LEN, L.POST[1], L.BAR_Z1 - L.BAR_Z0), color=L.HEAD_GREY),
    Cylinder("needle", pos=(L.X_SEAM, L.Y_NEEDLE, L.TOP + (L.NEEDLE_TIP + L.BAR_Z0) / 2), radius=L.NEEDLE_R,
             height=L.BAR_Z0 - L.NEEDLE_TIP, color=L.NEEDLE_STEEL),
]

_DX, _DY = L.DISH_HALF[0] + L.DISH_WALL / 2, L.DISH_HALF[1] + L.DISH_WALL / 2  # centre to the walls' centre lines
_WALL_Z = L.TOP + L.DISH_RIM / 2
DISH = [
    Box("dish_floor", pos=(L.DISH_X, L.DISH_Y, L.TOP + L.DISH_FLOOR_T / 2),
        size=(L.DISH_INNER[0], L.DISH_INNER[1], L.DISH_FLOOR_T), color=L.DISH_BLACK),
    Box("dish_wall_front", pos=(L.DISH_X + _DX, L.DISH_Y, _WALL_Z),
        size=(L.DISH_WALL, L.DISH_INNER[1] + 2 * L.DISH_WALL, L.DISH_RIM), color=L.DISH_BLACK),
    Box("dish_wall_back", pos=(L.DISH_X - _DX, L.DISH_Y, _WALL_Z),
        size=(L.DISH_WALL, L.DISH_INNER[1] + 2 * L.DISH_WALL, L.DISH_RIM), color=L.DISH_BLACK),
    Box("dish_wall_left", pos=(L.DISH_X, L.DISH_Y + _DY, _WALL_Z),
        size=(L.DISH_INNER[0], L.DISH_WALL, L.DISH_RIM), color=L.DISH_BLACK),
    Box("dish_wall_right", pos=(L.DISH_X, L.DISH_Y - _DY, _WALL_Z),
        size=(L.DISH_INNER[0], L.DISH_WALL, L.DISH_RIM), color=L.DISH_BLACK),
]
OBJECTS = [WORK, *PIN_OBJECTS, *NEEDLE_HEAD, *DISH]

_TRACKER = S.Tracker()
_MODEL = offline_model.OfflineWork()  # offline checks only (see _observe)
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


# -- the offline checker's gate --------------------------------------------------------------------------------------
# OFFLINE CHECKER ONLY. Our NumPy checker (tools/offline/check_task.py, no physics) never moves mechkit's joints and
# does not carry pins that ride the plate, so on ITS states the scorer is fed the feed and the pins from
# offline_model.py. The gate is mechkit's own marker: the checker's work state says source "offline", every state the
# simulator builds says "sim". The simulator never takes this branch.
def is_offline_checker_state(work) -> bool:
    """True only for the work states our offline checker builds (mechkit's source "offline")."""
    return getattr(work, "source", "sim") == "offline"


def _offline_checker_states(arms, step: bool) -> tuple[float, dict]:
    """OFFLINE CHECKER ONLY: the feed and the pins from the kinematic model, stepped once per control step (from
    success(); score() reads it without stepping)."""
    return _MODEL.step(arms) if step else _MODEL.state()


def _feed(work) -> float:
    """The work's feed (m) from mechkit's joint reading; NaN if the state is broken."""
    try:
        if not np.all(np.isfinite(np.asarray(work.origin, dtype=float))):
            raise ValueError("origin")
        return float(work.joints["feed"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return math.nan


def _observe(objects, arms, step: bool) -> tuple[float, dict]:
    """The feed and the pins the scorer judges: in the simulator, mechkit's joint reading and the pins' own states;
    on our offline checker's states only, the offline model's (see the gate above)."""
    work = objects.get("work")
    if is_offline_checker_state(work):
        return _offline_checker_states(arms, step)
    return _feed(work), {n: objects.get(n) for n in L.PINS}


def success(objects) -> bool:
    """The seam complete (fed 160 mm or more), all three pins pulled in time (30-70 mm before the needle, never
    closer) and in the dish, both grippers let go, no strike, no pin knocked loose, nothing moved unheld; held still
    1 s. This runs first every
    control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _MODEL.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    feed, pins = _observe(objects, arms, step=True)
    result = _TRACKER.update(feed, pins, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "source": getattr(objects.get("work"), "source", None)}
    try:
        row["pin_pos"] = {n: np.asarray(p.pos, dtype=float) for n, p in pins.items()}
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a pin ever came within 6 mm of the needle line under the needle head (an upright pin's centre within
    15 mm), left the plate with no gripper holding it, or fell off the table, or the work moved more than 5 mm in all
    while no gripper held its handle. Else 1.0 for the seam complete (fed 160 mm), all three pins pulled in time
    (d 30-70 mm, never closer) and in the dish, both grippers let go; 0.75 for the seam complete with all three
    pulled in time; 0.5 for two pulled in time; 0.25 for one pulled in time and the work fed 45 mm; 0 otherwise."""
    arms = rigstate.arms()
    feed, pins = _observe(objects, arms, step=False)
    return _TRACKER.result(feed, pins, arms).score


# -- the scripted demo -------------------------------------------------------------------------------------------
# Both grippers close their fingers along x (left yaw -pi/2, right +pi/2: the signs joint6 allows), across the
# handle and the pins, so the fingers stay out of the needle head's way along the seam. Finger targets are minus
# half the commanded gap: 8 mm narrower than the part.
OPEN = -0.04695
HANDLE_GRIP = -(L.HANDLE[0] - 0.008) / 2  # 22 mm on the 30 mm handle
PIN_GRIP = -(2 * L.PIN_R - 0.008) / 2  # 10 mm on the 18 mm pin
YAW = {"left": -math.pi / 2, "right": math.pi / 2}
GRIP_Z = L.TOP + 0.045  # the finger tips 10 mm above the plate (on the handle and on a pin: 22 mm of pad on it)
ENTRY_Z = L.TOP + 0.075  # open tips 6 mm over a pin's top, level with the handle's: the last 3 cm go slowly
TRAVEL_Z = L.TOP + 0.110  # over the handle with the fingers open
CARRY_Z = L.TOP + 0.100  # a carried pin's bottom 71 mm above the table: 49 mm over the plate
DISH_ENTRY_Z = L.TOP + 0.060
DISH_DROP_Z = L.TOP + 0.035  # the pin's bottom 2 mm over the dish floor, the finger tips 7 mm over its rim
D_PULL = 0.055  # each pin is pulled with d 55 mm: 25 mm inside the 30-70 mm window's near end, 15 mm inside its far end
PULL = (L.X_SEAM, L.Y_NEEDLE + D_PULL)
DISH_LANE_Y = PULL[1]  # the right arm moves along x at the pull's y (clear of the needle head), then along y
FEED_END = 0.178  # 8 mm past FEED_DONE, 2 mm short of the rail's end
LIFT_MARGIN_S = 0.2  # the left arm feeds this long after the right arm has lifted the pin clear


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _spot(pin: str) -> tuple[float, float]:
    """Where the demo stands a pin in the dish."""
    return (L.DISH_X + L.DISH_SPOTS[pin], L.DISH_Y)


def _left_approach() -> list[Waypoint]:
    """Left arm: over the handle, down round it, grip it, settle."""
    yaw = YAW["left"]
    x, y0 = L.X_SEAM, L.HANDLE_Y
    return [
        Waypoint(2.5, (x, y0, TRAVEL_Z), OPEN, "over the handle", yaw),
        Waypoint(1.0, (x, y0, ENTRY_Z), OPEN, "down beside it", yaw),
        Waypoint(1.5, (x, y0, GRIP_Z), OPEN, "the last 3 cm slowly", yaw),
        Waypoint(0.6, (x, y0, GRIP_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (x, y0, GRIP_Z), HANDLE_GRIP, "grip the handle", yaw),
        Waypoint(0.5, (x, y0, GRIP_Z), HANDLE_GRIP, "settle", yaw),
    ]


HANDLE_HELD_AT = sum(w.duration for w in _left_approach())  # 7.6 s: the left gripper closed on the handle, settled
FIRST_PULL_MARGIN_S = 0.5  # pin 1 is lifted only this long after that (the real fingers lag the command)


def _pull_and_drop(pin: str, came_from: str | None) -> list[Waypoint]:
    """Right arm: from ready, or back from the dish along y to the lane (turning the fingers to close along x) and
    along x, to over the pull point; down slowly round the pin; grip; straight up; along x to its place's x; along y
    over its place in the dish, turning the fingers to close along y; down; let go; up. Every turn happens beside the
    needle head (x 0.285 or less, the head's posts start at 0.328), never over the work."""
    x, y = PULL
    sx, sy = _spot(pin)
    turn, down = YAW["right"], 0.0  # the fingers close along x over the work, along y over the dish
    wps = []
    if came_from is None:
        wps.append(Waypoint(2.0, (x, y, CARRY_Z), OPEN, "over the pull point", turn))
    else:
        px, _ = _spot(came_from)
        wps += [Waypoint(1.5, (px, DISH_LANE_Y, CARRY_Z), OPEN, "back along y to the lane, turning", turn),
                Waypoint(1.3, (x, y, CARRY_Z), OPEN, "along x over the pull point", turn)]
    wps += [
        Waypoint(0.6, (x, y, ENTRY_Z), OPEN, "down beside the pin", turn),
        Waypoint(1.5, (x, y, GRIP_Z), OPEN, "the last 3 cm slowly", turn),
        Waypoint(0.5, (x, y, GRIP_Z), OPEN, "settle", turn),
        Waypoint(1.5, (x, y, GRIP_Z), PIN_GRIP, "grip the pin", turn),
    ]
    settle = 0.3
    if came_from is None:  # pin 1: hold it in its pocket until the other gripper has the handle
        settle = max(settle, HANDLE_HELD_AT + FIRST_PULL_MARGIN_S - sum(w.duration for w in wps))
    wps += [
        Waypoint(settle, (x, y, GRIP_Z), PIN_GRIP, "settle" if came_from else "settle, until the plate is held",
                 turn),
        Waypoint(1.2, (x, y, CARRY_Z), PIN_GRIP, "pull it straight up", turn),
        Waypoint(1.3, (sx, DISH_LANE_Y, CARRY_Z), PIN_GRIP, "along x to its place's x", turn),
        Waypoint(1.5, (sx, sy, CARRY_Z), PIN_GRIP, "along y over its place in the dish, turning", down),
        Waypoint(0.8, (sx, sy, DISH_ENTRY_Z), PIN_GRIP, "down toward the dish", down),
        Waypoint(1.2, (sx, sy, DISH_DROP_Z), PIN_GRIP, "lower it into the dish", down),
        Waypoint(0.5, (sx, sy, DISH_DROP_Z), PIN_GRIP, "settle", down),
        Waypoint(1.5, (sx, sy, DISH_DROP_Z), OPEN, "let go", down),
        Waypoint(0.8, (sx, sy, CARRY_Z), OPEN, "up out of the dish", down),
    ]
    return wps


def _right() -> list[Waypoint]:
    plan, prev = [], None
    for pin in L.PINS:
        plan += _pull_and_drop(pin, prev)
        prev = pin
    return plan


def _lift_ends() -> list[float]:
    """When the right arm has each pin pulled straight up (the end of 'pull it straight up')."""
    out, t = [], 0.0
    for w in _right():
        t += w.duration
        if w.label == "pull it straight up":
            out.append(t)
    return out


def _left() -> list[Waypoint]:
    """Left arm: over the handle, down round it, grip; after each pull, feed the plate by its handle (50 mm, 50 mm,
    then to 178 mm), holding still in between; let go and rise."""
    yaw = YAW["left"]
    x, y0 = L.X_SEAM, L.HANDLE_Y
    plan = _left_approach()
    # (the feed after each move, its duration): each brings the next pin to d 55 mm; the last completes the seam
    feeds = [(L.PIN_D0["pin_2"] - D_PULL, 2.5), (L.PIN_D0["pin_3"] - D_PULL, 2.5), (FEED_END, 3.0)]
    for (feed, dur), lifted in zip(feeds, _lift_ends()):
        wait = lifted + LIFT_MARGIN_S - _span(plan)
        plan.append(Waypoint(max(wait, 0.05), (x, y0 - _fed(plan), GRIP_Z), HANDLE_GRIP,
                             "hold the plate still while the pin is pulled", yaw))
        plan.append(Waypoint(dur, (x, y0 - feed, GRIP_Z), HANDLE_GRIP, f"feed the plate to {feed * 1000:.0f} mm",
                             yaw))
    plan += [
        Waypoint(0.6, (x, y0 - FEED_END, GRIP_Z), HANDLE_GRIP, "settle", yaw),
        Waypoint(1.5, (x, y0 - FEED_END, GRIP_Z), OPEN, "let go of the handle", yaw),
        Waypoint(1.2, (x, y0 - FEED_END, TRAVEL_Z), OPEN, "up off the handle", yaw),
    ]
    return plan


def _fed(plan) -> float:
    """How far the left arm's plan has fed the plate so far (its last waypoint's y)."""
    return L.HANDLE_Y - plan[-1].pos[1]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
