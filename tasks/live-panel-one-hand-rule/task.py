"""On the station table: a live electrical panel (a dark 160 x 160 mm mounting plate with a yellow border, fixed) in
front of the left arm, with a porcelain fuse base at its centre holding a red (blown) cartridge fuse, 20 mm across
and 50 mm tall, between two copper busbars; a green spare fuse standing in a blue cup in front of the right arm; a
white hand-off pad on the midline, outside the panel; a black discard tray in front of the left arm; and a grey
parking square in front of each arm's base. One-hand rule: the first gripper to go inside the panel is the panel
hand; the other gripper may never go inside it, and whenever the panel hand is inside, the other gripper must be in
its own parking square. Only the right arm reaches the spare and only the left arm reaches the socket and the tray
(straight-down grips), so the right arm stages the spare on the pad and parks before the left arm goes in.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

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
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s: green fuse seated, red fuse in the tray, both grippers out of the panel.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "The panel inside the yellow outline is live. Replace the red fuse in its socket with the green spare from the "
    "blue cup, and put the red fuse in the black tray; the spare may be set down on the white pad. Only one gripper "
    "may ever go inside the yellow outline, and while it is inside, the other gripper must wait in its own grey "
    "parking square, less than 10 cm above it."
)


def _walls(prefix: str, x: float, y: float, z0: float, inner: float, wall: float, height: float,
           color) -> list[Box]:
    """Four fixed walls around an ``inner`` x ``inner`` opening centred at (x, y), standing on height ``z0``."""
    z = z0 + height / 2
    o = inner / 2 + wall / 2
    return [
        Box(f"{prefix}_wall_front", pos=(x + o, y, z), size=(wall, inner + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_back", pos=(x - o, y, z), size=(wall, inner + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_left", pos=(x, y + o, z), size=(inner, wall, height), color=color),
        Box(f"{prefix}_wall_right", pos=(x, y - o, z), size=(inner, wall, height), color=color),
    ]


def _panel() -> list[Box]:
    """The mounting plate, its yellow border (four strips on its edges), the fuse base and the two busbars."""
    px, py, s, w = L.PANEL_X, L.PANEL_Y, L.PANEL_SIZE, L.BORDER_W
    zb = L.PLATE_TOP + L.BORDER_H / 2
    e = L.PANEL_HALF - w / 2
    return [
        Box("panel_plate", pos=(px, py, L.TOP + L.PLATE_H / 2), size=(s, s, L.PLATE_H), color=L.PLATE_GREY),
        Box("panel_edge_front", pos=(px + e, py, zb), size=(w, s, L.BORDER_H), color=L.PANEL_YELLOW),
        Box("panel_edge_back", pos=(px - e, py, zb), size=(w, s, L.BORDER_H), color=L.PANEL_YELLOW),
        Box("panel_edge_left", pos=(px, py + e, zb), size=(s - 2 * w, w, L.BORDER_H), color=L.PANEL_YELLOW),
        Box("panel_edge_right", pos=(px, py - e, zb), size=(s - 2 * w, w, L.BORDER_H), color=L.PANEL_YELLOW),
        *_walls("socket", L.SOCKET_X, L.SOCKET_Y, L.PLATE_TOP, L.SOCKET_INNER, L.SOCKET_WALL, L.SOCKET_HEIGHT,
                L.IVORY),
        Box("busbar_front", pos=(L.SOCKET_X + L.BUS_OFFSET, L.SOCKET_Y, L.PLATE_TOP + L.BUS_H / 2),
            size=(L.BUS_LEN, L.BUS_W, L.BUS_H), color=L.COPPER),
        Box("busbar_back", pos=(L.SOCKET_X - L.BUS_OFFSET, L.SOCKET_Y, L.PLATE_TOP + L.BUS_H / 2),
            size=(L.BUS_LEN, L.BUS_W, L.BUS_H), color=L.COPPER),
    ]


RED = Cylinder("red_fuse", pos=(L.RED_X, L.RED_Y, L.SOCKET_FLOOR + L.FUSE_HEIGHT / 2), radius=L.FUSE_RADIUS,
               height=L.FUSE_HEIGHT, mass=L.FUSE_MASS, color=L.RED)
GREEN = Cylinder("green_fuse", pos=(L.CUP_X, L.CUP_Y, L.TOP + L.FUSE_HEIGHT / 2), radius=L.FUSE_RADIUS,
                 height=L.FUSE_HEIGHT, mass=L.FUSE_MASS, color=L.GREEN)
CUP = _walls("cup", L.CUP_X, L.CUP_Y, L.TOP, L.CUP_INNER, L.CUP_WALL, L.CUP_HEIGHT, L.CUP_BLUE)
HANDOFF = Box("handoff_pad", pos=(L.HANDOFF_X, L.HANDOFF_Y, L.TOP + L.HANDOFF_SIZE[2] / 2), size=L.HANDOFF_SIZE,
              color=L.WHITE)
TRAY = _walls("tray", L.TRAY_X, L.TRAY_Y, L.TOP, L.TRAY_INNER, L.TRAY_WALL, L.TRAY_HEIGHT, L.TRAY_BLACK)
PARKING = [Box(f"park_{side}", pos=(L.PARK_X, L.PARK_Y[side], L.TOP + L.PARK_H / 2),
               size=(L.PARK_SIZE, L.PARK_SIZE, L.PARK_H), color=L.PARK_GREY) for side in ("left", "right")]
OBJECTS = [*_panel(), RED, *CUP, GREEN, HANDOFF, *TRAY, *PARKING]

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


def success(objects) -> bool:
    """Green fuse seated in the socket, red fuse in the discard tray, both grippers out of the panel, the one-hand
    rule kept all along; held still 1 s. This runs first every control step, so it also feeds the tracker (with the
    arms rigstate read for this step)."""
    red, green = objects["red_fuse"], objects["green_fuse"]
    arms = rigstate.arms()
    result = _TRACKER.update(red, green, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "red_pos": red.pos, "green_pos": green.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a second gripper ever went inside the panel, if the panel hand was ever inside it while the other
    gripper was out of its parking square, or if a fuse left the table. Else 1.0 green fuse seated, red fuse in the
    discard tray, both grippers out of the panel; 0.75 green seated and the red fuse clear of the panel; 0.5 red
    fuse out of the socket and the green fuse inside the panel; 0.25 red fuse in the discard tray; 0 otherwise."""
    return _TRACKER.result(objects["red_fuse"], objects["green_fuse"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the 20 mm fuse.
OPEN = -0.04695
GRIP = -(2 * L.FUSE_RADIUS - 0.008) / 2
YAW = 0.0  # both arms close the fingers along y (the bearing of every grasp point is within 60 deg of x)
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_DEPTH = 0.046  # TCP above a held fuse's bottom: the tips 33 mm up the fuse, its top 17 mm held between the pads
HOVER = L.TOP + 0.12  # over the socket this is inside the panel's live zone (below 15 cm)
CARRY = L.TOP + 0.12  # a carried fuse's bottom 74 mm above the table: clear of every wall
WAIT_Z = L.TOP + 0.14  # the left arm waits (and ends) up here over the discard tray, outside the panel
PARK_Z = L.TOP + 0.05  # parked: TCP 5 cm above the table over the square (the square's zone ends at 10 cm)
CUP_GRIP_Z = L.TOP + GRIP_DEPTH  # tips 13 mm above the cup's 20 mm rim
SOCKET_GRIP_Z = L.SOCKET_FLOOR + GRIP_DEPTH  # tips 8 mm above the socket's 25 mm rim
SOCKET_ENTRY_Z = L.SOCKET_RIM + 0.015 + GRIP_DEPTH  # the spare's bottom 15 mm above the rim
SOCKET_SET_Z = L.SOCKET_FLOOR + 0.002 + GRIP_DEPTH  # let go 2 mm above the floor: tips 10 mm above the rim
HANDOFF_SET_Z = L.HANDOFF_TOP + 0.002 + GRIP_DEPTH  # let go 2 mm above the pad
HANDOFF_GRIP_Z = L.HANDOFF_TOP + GRIP_DEPTH
TRAY_SET_Z = L.TOP + 0.005 + GRIP_DEPTH  # let go 5 mm above the tray floor: tips 13 mm above its 25 mm rim
ENTRY = 0.020  # the last 2 cm of a set-down are lowered slowly


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _right() -> list[Waypoint]:
    """Right arm: take the green spare out of its cup, set it upright on the hand-off pad, then go to its parking
    square and stay there for the rest of the trial."""
    cx, cy = L.CUP_X, L.CUP_Y
    hx, hy = L.HANDOFF_X, L.HANDOFF_Y
    kx, ky = L.PARK_X, L.PARK_Y["right"]
    return [
        Waypoint(2.0, (cx, cy, HOVER), OPEN, "over the spare in its cup", YAW),
        Waypoint(1.5, (cx, cy, CUP_GRIP_Z), OPEN, "down beside it", YAW),
        Waypoint(0.6, (cx, cy, CUP_GRIP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (cx, cy, CUP_GRIP_Z), GRIP, "grip the spare", YAW),
        Waypoint(1.5, (cx, cy, CARRY), GRIP, "lift it out of the cup", YAW),
        Waypoint(2.0, (hx, hy, CARRY), GRIP, "carry it over the hand-off pad", YAW),
        Waypoint(1.0, (hx, hy, HANDOFF_SET_Z + ENTRY), GRIP, "down to 2 cm above", YAW),
        Waypoint(0.5, (hx, hy, HANDOFF_SET_Z + ENTRY), GRIP, "settle", YAW),
        Waypoint(1.5, (hx, hy, HANDOFF_SET_Z), GRIP, "lower the last 2 cm slowly", YAW),
        Waypoint(0.5, (hx, hy, HANDOFF_SET_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (hx, hy, HANDOFF_SET_Z), OPEN, "let go on the pad", YAW),
        Waypoint(1.2, (hx, hy, HOVER), OPEN, "up", YAW),
        Waypoint(2.0, (kx, ky, HOVER), OPEN, "over the parking square", YAW),
        Waypoint(1.2, (kx, ky, PARK_Z), OPEN, "down into the parking square", YAW),
    ]


RIGHT_PARKED_S = _span(_right())  # the right gripper is down in its parking square
WAIT_MARGIN_S = 1.0  # the left arm moves toward the panel only this long after that (the right arm's tracking lag)


def _left() -> list[Waypoint]:
    """Left arm (the panel hand): wait over the discard tray until the right arm is parked; then pull the red fuse
    out of the socket, drop it in the tray, take the spare from the pad, lower it into the socket, let go and leave
    the panel."""
    sx, sy = L.SOCKET_X, L.SOCKET_Y
    tx, ty = L.TRAY_X, L.TRAY_Y
    hx, hy = L.HANDOFF_X, L.HANDOFF_Y
    aside = Waypoint(2.0, (tx, ty, WAIT_Z), OPEN, "aside, over the discard tray", YAW)
    wait = RIGHT_PARKED_S + WAIT_MARGIN_S - aside.duration
    return [
        aside,
        Waypoint(wait, (tx, ty, WAIT_Z), OPEN, "wait until the right gripper is parked", YAW),
        Waypoint(2.0, (sx, sy, HOVER), OPEN, "into the panel, over the red fuse", YAW),
        Waypoint(1.5, (sx, sy, SOCKET_GRIP_Z), OPEN, "down beside it", YAW),
        Waypoint(0.6, (sx, sy, SOCKET_GRIP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (sx, sy, SOCKET_GRIP_Z), GRIP, "grip the red fuse", YAW),
        Waypoint(1.5, (sx, sy, CARRY), GRIP, "pull it out of the socket", YAW),
        Waypoint(2.0, (tx, ty, CARRY), GRIP, "carry it out of the panel, over the tray", YAW),
        Waypoint(1.0, (tx, ty, TRAY_SET_Z + ENTRY), GRIP, "down to 2 cm above", YAW),
        Waypoint(0.5, (tx, ty, TRAY_SET_Z + ENTRY), GRIP, "settle", YAW),
        Waypoint(1.2, (tx, ty, TRAY_SET_Z), GRIP, "lower it into the tray", YAW),
        Waypoint(0.5, (tx, ty, TRAY_SET_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (tx, ty, TRAY_SET_Z), OPEN, "let it drop into the tray", YAW),
        Waypoint(1.0, (tx, ty, HOVER), OPEN, "up", YAW),
        Waypoint(2.0, (hx, hy, HOVER), OPEN, "over the spare on the pad", YAW),
        Waypoint(1.5, (hx, hy, HANDOFF_GRIP_Z), OPEN, "down beside it", YAW),
        Waypoint(0.6, (hx, hy, HANDOFF_GRIP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (hx, hy, HANDOFF_GRIP_Z), GRIP, "grip the spare", YAW),
        Waypoint(1.5, (hx, hy, CARRY), GRIP, "lift it off the pad", YAW),
        Waypoint(2.0, (sx, sy, CARRY), GRIP, "carry it into the panel, over the socket", YAW),
        Waypoint(1.2, (sx, sy, SOCKET_ENTRY_Z), GRIP, "down to 15 mm above the rim", YAW),
        Waypoint(0.6, (sx, sy, SOCKET_ENTRY_Z), GRIP, "settle", YAW),
        Waypoint(2.0, (sx, sy, SOCKET_SET_Z), GRIP, "lower it into the socket slowly", YAW),
        Waypoint(0.5, (sx, sy, SOCKET_SET_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (sx, sy, SOCKET_SET_Z), OPEN, "let go", YAW),
        Waypoint(1.2, (sx, sy, HOVER), OPEN, "up", YAW),
        Waypoint(1.5, (tx, ty, WAIT_Z), OPEN, "out of the panel, back over the tray", YAW),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
