"""On the station table: a surface grinder's magnetic chuck (fixed, 200 x 100 x 25 mm, long side along x) centred
27 cm in front of the arm bases and 2 cm right of the midline, its face of steel pole pieces crossed by three 6 mm
brass lines 60 mm apart, with a 5 mm steel rail along its left long edge; a loose blue steel part (40 x 30 x 20 mm,
188 g) on the table to the left, its long side along y; and four loose black steel blocking pieces on the table to
the right: two lower than the part (40 x 18 x 15 mm) and two taller than it whichever way they lie (40 x 24 x 28 mm).
Task: the part flat on the chuck, lengthwise against the rail, across a brass line, then a low piece against each
end; no piece taller than the part left on the chuck; let go of everything.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

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
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s (the part and both blocking pieces), with every counted item let go (no finger zone reaching it).
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Set the blue steel part flat on the magnetic chuck for grinding: lengthwise against the rail along the "
    "chuck's left edge, straddling one of the brass lines. Then block it in: put a blocking piece against each end "
    "of the part, using only pieces lower than the part so the grinding wheel passes over them. Let go of "
    "everything when done."
)


def _chuck() -> list[Box]:
    """The chuck, built from its full-height face strips: steel pole pieces and the brass lines between them."""
    out = []
    for i, (x0, x1) in enumerate(L.pole_spans()):
        out.append(Box(f"chuck_pole_{i + 1}", pos=((x0 + x1) / 2, L.CHUCK_Y, L.TOP + L.CHUCK_H / 2),
                       size=(x1 - x0, L.CHUCK_W, L.CHUCK_H), color=L.POLE_STEEL))
    for i, x in enumerate(L.LINES_X):
        out.append(Box(f"chuck_brass_{i + 1}", pos=(x, L.CHUCK_Y, L.TOP + L.CHUCK_H / 2),
                       size=(L.LINE_W, L.CHUCK_W, L.CHUCK_H), color=L.BRASS))
    return out


CHUCK = _chuck()
RAIL = Box("chuck_rail", pos=(L.CHUCK_X, L.RAIL_Y, L.FACE_Z + L.RAIL_H / 2),
           size=(L.CHUCK_X1 - L.CHUCK_X0, L.RAIL_W, L.RAIL_H), color=L.RAIL_STEEL)
PART = Box("part", pos=(L.PART_X0, L.PART_Y0, L.TOP + L.PART_SIZE[2] / 2), size=L.PART_SIZE, mass=L.PART_MASS,
           color=L.LAYOUT_BLUE, yaw=L.PART_YAW0)
PIECES = [
    Box(name, pos=(x, y, L.TOP + L.SIZES[name][2] / 2), size=L.SIZES[name],
        mass=(L.LOW_MASS if kind == "low" else L.TALL_MASS), color=L.BLACK_OXIDE)
    for name, kind, x, y in L.PIECES
]
OBJECTS = [*CHUCK, RAIL, PART, *PIECES]

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


def _loose(objects) -> dict:
    return {n: objects[n] for n in L.LOOSE if n in objects}


def success(objects) -> bool:
    """Part flat against the rail across a brass line, a low piece against each end, no piece taller than the part
    on the chuck, nothing held; the part and both pieces still for 1 s. This runs first every control step, so it
    also feeds the tracker (with the arms rigstate read for this step)."""
    states = _loose(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(states, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in states.items():
        row[f"{n}_pos"] = getattr(s, "pos", None)
    row["part_quat"] = getattr(states.get("part"), "quat_xyzw", None)
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a piece taller than the part (top over 21 mm above the face) is on the chuck. Else 0.25 for the part
    flat, lengthwise and within 3 mm of the rail (let go); with that, 0.25 more if a brass line is under it with
    8 mm of steel each side, and 0.25 for each end with a low piece flat against it (within 3 mm, square within
    10 degrees, covering 10 mm of the end face, let go)."""
    return _TRACKER.result(_loose(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PART_GRIP = -(L.PART_SIZE[0] - 0.008) / 2  # across the part's 40 mm length, pads on its two 30 x 20 mm ends
PIECE_WIDE = -0.022  # 44 mm apart over a piece: 13 mm each side of an 18 mm piece, clear of its neighbours
PIECE_GRIP = -(L.LOW_SIZE[1] - 0.008) / 2  # across a low piece's 18 mm width
PIECE_LET_GO = -0.013  # 26 mm apart: 4 mm each side, so the pad on the rail's side stays 2 mm short of the rail
YAW_PICK, YAW_PLACE = 0.0, -math.pi / 2  # the left arm turns the part from along y to along x (the rail)
YAW_PIECE = 0.0  # the right arm closes across a piece's width (y); the piece keeps its heading
CARRY_Z = L.TOP + 0.10
PARK_Z = L.TOP + 0.15
ENTRY = 0.020  # the last 20 mm down are taken slowly
PART_UP = 0.020  # TCP 20 mm above the part's bottom: pad tips 7 mm up, pads on its upper 13 mm
PIECE_UP = 0.0195  # TCP 19.5 mm above a piece's bottom: pad tips 6.5 mm up, clear above the 5 mm rail
LIFT = 0.0005  # let go with the bottom 0.5 mm above the face
# Where the part goes: centred over the middle brass line, its side 2.8 mm short of the rail. The arms are joint
# springs (80/80/80/10/10/10 N m/rad) and the controller does not carry a payload's weight: the compliance model
# (J K^-1 J^T) puts the left arm, holding the 188 g part at the place point, (-1.5, +1.7, -4.8) mm off its command
# at the tool point and turns the wrist 0.6 deg, which brings the part's lower edge on the rail's side 1.9 mm nearer
# the rail. So on the GPU it passes the rail's top about 0.9-1.2 mm clear and lands that far from the rail (2.8 mm
# here, with no sag; 'against' allows 3 mm). The heatsink task's GPU run landed (-1.8, +1.6) mm off its command
# against a modelled (-1.9, +1.9). Aiming closer would risk the part's lower edge catching the rail's top on the way
# down. Read rail_gap_near_mm and rail_gap_far_mm in TASK_TRACE.
RAIL_AIM = 0.0028
PART_X = L.LINES_X[1]
PART_Y = L.RAIL_FACE_Y - L.PART_HALF[1] - RAIL_AIM
# The low pieces cover the 18 mm of the end face farthest from the rail, 14 mm short of it (the end face is 30 mm
# deep), leaving room for the gripper's rail-side pad.
PIECE_Y = L.RAIL_FACE_Y - 0.014 - L.LOW_SIZE[1] / 2
# Gaps aimed at each end. With 85 g the right arm sags (-0.7 to -0.8, -0.8 to -0.9, -2.0 to -2.5) mm at a piece's
# lower edge at the two places, and the part's lower edge lands 1.7 mm nearer the arms than here, so on the GPU the
# near gap closes by about 1.0 mm (2.3 -> 1.3) and the far one opens by about 0.9 mm (1.0 -> 1.9); 'against' allows
# 3 mm. Aiming the near piece closer would risk its lower edge catching the part's top on the way down. Read
# near_gap_mm and far_gap_mm in TASK_TRACE.
GAP_NEAR, GAP_FAR = 0.0023, 0.0010
FAR_X = PART_X + L.PART_HALF[0] + GAP_FAR + L.LOW_SIZE[0] / 2
NEAR_X = PART_X - L.PART_HALF[0] - GAP_NEAR - L.LOW_SIZE[0] / 2
PART_PICK_Z = L.TOP + PART_UP
PART_PLACE_Z = L.FACE_Z + LIFT + PART_UP
PIECE_PICK_Z = L.TOP + PIECE_UP
PIECE_PLACE_Z = L.FACE_Z + LIFT + PIECE_UP
LEFT_PARK = (0.18, 0.30, PARK_Z)
RIGHT_PARK = (0.18, -0.30, PARK_Z)
RIGHT_WAIT = (0.20, -0.26, CARRY_Z)  # holding the first piece 26 cm right of the midline while the part goes down
RIGHT_GO = 20.0  # s: the right arm leaves the waiting point once the left arm has let go and lifted (19.3 s)


def _left() -> list[Waypoint]:
    """Left arm: pick the part by its ends, turn it lengthwise on the way, set it down by the rail across the
    middle brass line, let go, park."""
    x0, y0 = L.PART_X0, L.PART_Y0
    return [
        Waypoint(2.0, (x0, y0, CARRY_Z), OPEN, "over the part", YAW_PICK),
        Waypoint(1.5, (x0, y0, PART_PICK_Z + ENTRY), OPEN, "down to 2 cm above", YAW_PICK),
        Waypoint(1.5, (x0, y0, PART_PICK_Z), OPEN, "slowly down round its ends", YAW_PICK),
        Waypoint(0.5, (x0, y0, PART_PICK_Z), OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (x0, y0, PART_PICK_Z), PART_GRIP, "grip the part by its ends", YAW_PICK),
        Waypoint(0.5, (x0, y0, PART_PICK_Z), PART_GRIP, "settle the grip", YAW_PICK),
        Waypoint(1.5, (x0, y0, CARRY_Z), PART_GRIP, "lift", YAW_PICK),
        Waypoint(2.5, (PART_X, PART_Y, CARRY_Z), PART_GRIP, "carry over the chuck, turning it lengthwise", YAW_PLACE),
        Waypoint(1.5, (PART_X, PART_Y, PART_PLACE_Z + ENTRY), PART_GRIP, "down to 2 cm above the face", YAW_PLACE),
        Waypoint(0.8, (PART_X, PART_Y, PART_PLACE_Z + ENTRY), PART_GRIP, "settle", YAW_PLACE),
        Waypoint(2.0, (PART_X, PART_Y, PART_PLACE_Z), PART_GRIP, "lower the last 2 cm slowly", YAW_PLACE),
        Waypoint(1.5, (PART_X, PART_Y, PART_PLACE_Z), OPEN, "let go", YAW_PLACE),
        Waypoint(0.5, (PART_X, PART_Y, PART_PLACE_Z), OPEN, "settle", YAW_PLACE),
        Waypoint(1.5, (PART_X, PART_Y, CARRY_Z), OPEN, "straight up", YAW_PLACE),
        Waypoint(2.0, LEFT_PARK, OPEN, "park to the left", YAW_PICK),
    ]


def _pick(x: float, y: float, approach: float) -> list[Waypoint]:
    return [
        Waypoint(approach, (x, y, CARRY_Z), PIECE_WIDE, "over a low blocking piece", YAW_PIECE),
        Waypoint(1.5, (x, y, PIECE_PICK_Z + ENTRY), PIECE_WIDE, "down to 2 cm above", YAW_PIECE),
        Waypoint(1.5, (x, y, PIECE_PICK_Z), PIECE_WIDE, "slowly down round its sides", YAW_PIECE),
        Waypoint(0.5, (x, y, PIECE_PICK_Z), PIECE_WIDE, "settle", YAW_PIECE),
        Waypoint(1.5, (x, y, PIECE_PICK_Z), PIECE_GRIP, "grip the piece", YAW_PIECE),
        Waypoint(0.5, (x, y, PIECE_PICK_Z), PIECE_GRIP, "settle the grip", YAW_PIECE),
        Waypoint(1.5, (x, y, CARRY_Z), PIECE_GRIP, "lift", YAW_PIECE),
    ]


def _place(x: float, label: str, carry: float) -> list[Waypoint]:
    y = PIECE_Y
    return [
        Waypoint(carry, (x, y, CARRY_Z), PIECE_GRIP, f"carry over the {label} end", YAW_PIECE),
        Waypoint(1.5, (x, y, PIECE_PLACE_Z + ENTRY), PIECE_GRIP, "down to 2 cm above the face", YAW_PIECE),
        Waypoint(0.8, (x, y, PIECE_PLACE_Z + ENTRY), PIECE_GRIP, "settle", YAW_PIECE),
        Waypoint(2.0, (x, y, PIECE_PLACE_Z), PIECE_GRIP, f"lower the last 2 cm slowly against the {label} end",
                 YAW_PIECE),
        Waypoint(1.5, (x, y, PIECE_PLACE_Z), PIECE_LET_GO, "let go", YAW_PIECE),
        Waypoint(0.5, (x, y, PIECE_PLACE_Z), PIECE_LET_GO, "settle", YAW_PIECE),
        Waypoint(1.5, (x, y, CARRY_Z), PIECE_LET_GO, "straight up", YAW_PIECE),
    ]


def _right() -> list[Waypoint]:
    """Right arm: pick the nearer low piece and wait well to the right; once the part is down and the left arm has
    lifted, set the piece against the part's far end, then the other low piece against its near end; park."""
    (x1, y1), (x2, y2) = L.STARTS["block_low_1"], L.STARTS["block_low_2"]
    first = _pick(x1, y1, 2.0) + [Waypoint(2.0, RIGHT_WAIT, PIECE_GRIP, "wait to the right", YAW_PIECE)]
    elapsed = sum(w.duration for w in first)
    first.append(Waypoint(RIGHT_GO - elapsed, RIGHT_WAIT, PIECE_GRIP, "hold until the part is down", YAW_PIECE))
    return (first + _place(FAR_X, "far", 2.0) + _pick(x2, y2, 2.5) + _place(NEAR_X, "near", 2.5)
            + [Waypoint(2.0, RIGHT_PARK, PIECE_LET_GO, "park to the right", YAW_PIECE)])


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
