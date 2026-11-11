"""On the station table, a herbarium mounting sheet at 1:1.5 (194.7 x 279.4 mm, white card, a 2 mm grey line round it
on the bench) lies portrait on the midline, its foot (bottom edge) 20.5 cm in front of the arm bases; its accession
barcode is stuck on in the top right corner. In front of the left arm lies the pressed specimen: a 200 x 70 x 10 mm
basswood board with the plant printed on top (brown roots at one end, a green stem and leaves, yellow flowers at
the other), roots toward +y. In front of the right arm lie the collection label (a white 67 x 47 x 12 mm pine tile
with printed lines) and the determinavit slip (a cream 60 x 20 x 12 mm pine tile with "Det." and a name line).
The task is to lay them out on the sheet as a herbarium mounts them; the rule and its tolerances are in scoring.py
and task.json (the instruction does not state them).

The three items are loose mechkit compounds (mechkit.py, next to this file): one rigid body each, so the printed
plant and text are part of them and the scorer can read each item's frame. The scorer also reads the arms (TCP,
pad-tip point, opening) through rigstate.py, which wraps the embodiment's step and reset from this file's import;
neither changes anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the items are built as compound bodies)
import rigstate  # noqa: E402
from scoring import Tracker, heading_deg, tilt_deg, valid  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Each item must be let go (both grippers' finger zones 20 mm or more from it) and still for 1 s, so the trial cannot
# end while a gripper still holds or touches one; then 1.0 must hold 1 s.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "This is a herbarium mounting sheet (the standard 11.5 x 16.5 in sheet, shown at 1:1.5 scale) lying portrait on "
    "the bench. Its foot, the bottom edge, is the edge nearest the arms, and the sheet is read upright from there. "
    "The accession barcode is already stuck on. Lay out this pressed specimen, its collection label and its "
    "determination slip on the sheet the way a herbarium mounts them, ready for gluing, then let go."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


W = L.OUTLINE_W
OUTLINE = [
    _span_box("outline_foot", L.SHEET_X0 - W, L.SHEET_X0, L.SHEET_Y0 - W, L.SHEET_Y1 + W, 0.0, L.OUTLINE_T,
              L.OUTLINE_GREY),
    _span_box("outline_head", L.SHEET_X1, L.SHEET_X1 + W, L.SHEET_Y0 - W, L.SHEET_Y1 + W, 0.0, L.OUTLINE_T,
              L.OUTLINE_GREY),
    _span_box("outline_right", L.SHEET_X0, L.SHEET_X1, L.SHEET_Y0 - W, L.SHEET_Y0, 0.0, L.OUTLINE_T, L.OUTLINE_GREY),
    _span_box("outline_left", L.SHEET_X0, L.SHEET_X1, L.SHEET_Y1, L.SHEET_Y1 + W, 0.0, L.OUTLINE_T, L.OUTLINE_GREY),
]
SHEET = _span_box("mounting_sheet", L.SHEET_X0, L.SHEET_X1, L.SHEET_Y0, L.SHEET_Y1, 0.0, L.SHEET_T, L.SHEET_WHITE)
BARCODE = _span_box("barcode", L.BARCODE_X0, L.BARCODE_X1, L.BARCODE_Y0, L.BARCODE_Y1, L.SHEET_T,
                    L.SHEET_T + L.BARCODE_T, L.BARCODE_WHITE)
BAR_Z0 = L.SHEET_T + L.BARCODE_T
BARS = [_span_box(f"barcode_bar_{i}", L.BAR_X0, L.BAR_X1, cy - w / 2, cy + w / 2, BAR_Z0, BAR_Z0 + L.BAR_T,
                  L.INK_BLACK) for i, (cy, w) in enumerate(L.BARS)]


def _tile(name, size, mass, start, color, marks) -> mk.Assembly:
    """A loose tile (label or slip) with its printed marks on top: (centre x, centre y, size x, size y)."""
    sx, sy, sz = size
    x, y, yaw = start
    shapes = [mk.box((0.0, 0.0, sz / 2), size)]
    shapes += [mk.box((cx, cy, sz + L.PRINT_T / 2), (wx, wy, L.PRINT_T), color=L.TEXT_GREY)
               for cx, cy, wx, wy in marks]
    return mk.assembly(name, origin=(x, y, L.TOP), yaw=math.radians(yaw), mass=mass, color=color, shapes=shapes)


LABEL_MARKS = [(cx, 0.025 - n / 2, w, n) for cx, n, w in L.LABEL_LINES]  # left-aligned lines, read along y
LABEL = _tile("label", L.LABEL_SIZE, L.LABEL_MASS, L.LABEL_START, L.LABEL_WHITE, LABEL_MARKS)
SLIP = _tile("slip", L.SLIP_SIZE, L.SLIP_MASS, L.SLIP_START, L.SLIP_CREAM, L.SLIP_MARKS)

_PLANT_COLOR = {"roots": L.ROOT_BROWN, "stem": L.LEAF_GREEN, "leaves": L.LEAF_GREEN, "flowers": L.FLOWER_YELLOW}
_BT = L.BOARD_SIZE[2]
SPECIMEN = mk.assembly(
    "specimen",
    origin=(L.SPEC_START[0], L.SPEC_START[1], L.TOP),
    yaw=math.radians(L.SPEC_START[2]),
    mass=L.SPEC_MASS,
    color=L.BOARD_BUFF,
    shapes=[mk.box((0.0, 0.0, _BT / 2), L.BOARD_SIZE)] + [
        mk.box((cx, cy, _BT + L.PLANT_T / 2), (wx, wy, L.PLANT_T), color=_PLANT_COLOR[part])
        for part, boxes in L.PLANT.items() for cx, cy, wx, wy in boxes
    ],
)
ITEMS = {"label": LABEL, "slip": SLIP, "specimen": SPECIMEN}
OBJECTS = [*OUTLINE, SHEET, BARCODE, *BARS, SPECIMEN, LABEL, SLIP]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and a teleport back to the start are
    two more signals)."""
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


def _items(objects) -> dict:
    return {n: objects.get(n) for n in L.ITEMS}


def _pose(st) -> dict:
    """An item's frame origin (mm), heading and tilt (deg) and speed (mm/s), for the trace."""
    if not valid(st):
        return {"origin_mm": None, "source": getattr(st, "source", None)}
    return {"origin_mm": [round(float(v) * 1000, 1) for v in st.origin],
            "heading_deg": round(heading_deg(st), 2), "tilt_deg": round(tilt_deg(st), 2),
            "speed_mm_s": round(float(sum(float(v) ** 2 for v in st.lin_vel) ** 0.5) * 1000, 1),
            "source": getattr(st, "source", None)}


def success(objects) -> bool:
    """Label in the bottom right corner 2-12 mm from both edges and square; slip directly above or left of it, 0-10 mm
    away, lined up within 10 mm; specimen on the sheet 2 mm in, roots toward the foot within 15 deg, clear of label,
    slip and barcode; each let go (finger zones >= 20 mm), face up within 5 deg, on the table (3 mm) and still for
    1 s; held 1 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate read)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    items = _items(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(items, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "score": result.score,
           "success": result.success, **result.details}
    row.update({f"{n}_pose": _pose(items[n]) for n in L.ITEMS})
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings=[round(float(arms["left"].opening), 3), round(float(arms["right"].opening), 3)])
        except (KeyError, AttributeError, TypeError, ValueError):
            row["rigstate_error"] = "bad snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.4 if the label is correct, + 0.2 if the slip is correct (only with the label correct), + 0.4 if the specimen
    is correct; an item counts only let go, face up and flat, on the table and still for 1 s. Read from the state at
    hand (the last step)."""
    return _TRACKER.result(_items(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part. Yaw 0 closes
# the fingers along world y.
def _grip(width: float) -> float:
    return -(width - 0.008) / 2


def _open(width: float, clearance: float) -> float:
    return -(width + 2 * clearance) / 2


SLIP_W = L.SLIP_SIZE[0]  # the slip is gripped across its 20 mm width (fingers along x, yaw 90)
LABEL_W = L.LABEL_SIZE[1]  # the label across its 67 mm length (fingers along y, yaw 0): no finger between it and
#                            the slip
SPEC_W = L.BOARD_SIZE[1]  # the specimen across its 70 mm board width at the middle (fingers along x at the start)
FINGERS = {
    "slip": (_open(SLIP_W, 0.008), _grip(SLIP_W)),  # open 36 mm, commanded 12 mm
    "label": (_open(LABEL_W, 0.007), _grip(LABEL_W)),  # open 81 mm, commanded 59 mm
    "specimen": (_open(SPEC_W, 0.008), _grip(SPEC_W)),  # open 86 mm, commanded 62 mm
}
PAD = 0.013  # the pad tips are this far below the tool point
GRIP_UP = 0.003 + PAD  # the TCP over the item's underside: pad tips 3 mm above it (9 of the tiles' 12 mm, 7 of 10)
DROP = 0.001  # each item is let go with its underside this far above the sheet
APPROACH = 0.030  # each final descent starts this far above its end point and takes 1.5 s
HOVER = L.TOP + 0.100  # the tiles carried with their underside 84 mm up
SPEC_HOVER = L.TOP + 0.120  # the board turned and carried with its underside 104 mm up
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.TOP + L.SHEET_T + DROP + GRIP_UP
# Gripped parts ended turned +0.6 to +1.7 degrees counter-clockwise of the commanded yaw in our GPU runs, so every
# place yaw is set 1.15 degrees clockwise (squareness allows 5 degrees, the roots 15).
YAW_BIAS = math.radians(-1.15)
# Heavier parts landed about 4 mm short of the place point along the carry in our GPU runs (the arm trails under
# load), while light mechkit parts (10-60 g, like these) landed 0.3-1.8 mm off aim in any direction. Every place
# point is aimed this far past its target along the carry, which keeps both cases inside every window: the label's
# right edge 8 mm in nominally, 4 mm if it lands 4 mm short, 9.8 mm if it lands 1.8 mm past (band 2-12 mm).
LAND_AHEAD = 0.001
# Trim (x, y) of each place point, for the GPU run: if an item lands off its target (TASK_TRACE: the label's
# right_edge_in_mm / foot_in_mm away from 7, the slip's gap_above_mm away from 6 or align_above_mm away from 0, the
# specimen's extent_mm away from x 255-455, y 10-80), move its trim by minus the offset.
PLACE_TRIM = {"slip": (0.0, 0.0), "label": (0.0, 0.0), "specimen": (0.0, 0.0)}
YAW = {  # (pick yaw, place yaw) of each grip (rad)
    "slip": (math.pi / 2, math.pi / 2 + YAW_BIAS),  # slip local x along world x: fingers on its long faces
    "label": (0.0, YAW_BIAS),
    # the board lies along y at the start (yaw 90): fingers along x at yaw -90 (joint 6 within reach), turned a
    # quarter turn anticlockwise to yaw 0 so the board ends at yaw 180, roots toward the foot
    "specimen": (-math.pi / 2, YAW_BIAS),
}
TARGET = {"slip": L.SLIP_TARGET, "label": L.LABEL_TARGET, "specimen": L.SPEC_TARGET[:2]}
PARK = {"left": (0.200, 0.300, L.TOP + 0.160), "right": (0.200, -0.300, L.TOP + 0.160)}


def place_point(name: str) -> tuple[float, float]:
    """The target, aimed LAND_AHEAD past it along the carry from the item's start, plus the trim."""
    sx, sy, _ = L.START[name]
    tx, ty = TARGET[name]
    d = math.hypot(tx - sx, ty - sy)
    ux, uy = (tx - sx) / d, (ty - sy) / d
    dx, dy = PLACE_TRIM[name]
    return tx + LAND_AHEAD * ux + dx, ty + LAND_AHEAD * uy + dy


def _pick(name: str, hover: float, first: float = 2.0) -> list[Waypoint]:
    """Over the item, down (slowly for the last 30 mm), grip it, lift it."""
    sx, sy, _ = L.START[name]
    op, gr = FINGERS[name]
    yaw = YAW[name][0]
    return [
        Waypoint(first, (sx, sy, HOVER), op, f"over the {name}", yaw),
        Waypoint(1.0, (sx, sy, PICK_Z + APPROACH), op, f"down to just above the {name}", yaw),
        Waypoint(1.5, (sx, sy, PICK_Z), op, f"slowly down round the {name}", yaw),
        Waypoint(0.5, (sx, sy, PICK_Z), op, "settle", yaw),
        Waypoint(1.5, (sx, sy, PICK_Z), gr, f"grip the {name}", yaw),
        Waypoint(0.3, (sx, sy, PICK_Z), gr, "settle the grip", yaw),
        Waypoint(1.5 if hover > HOVER else 1.2, (sx, sy, hover), gr, f"lift the {name}", yaw),
    ]


def _place(name: str, hover: float, carry: float = 2.0) -> list[Waypoint]:
    """Carry the item over its place, down (slowly for the last 30 mm), let go, straight up."""
    px, py = place_point(name)
    op, gr = FINGERS[name]
    yaw = YAW[name][1]
    return [
        Waypoint(carry, (px, py, hover), gr, f"carry the {name} over its place", yaw),
        Waypoint(0.5, (px, py, hover), gr, "settle over the place", yaw),
        Waypoint(1.0, (px, py, PLACE_Z + APPROACH), gr, "down to just above the sheet", yaw),
        Waypoint(1.5, (px, py, PLACE_Z), gr, f"slowly lay the {name} down", yaw),
        Waypoint(0.5, (px, py, PLACE_Z), gr, "settle", yaw),
        Waypoint(1.5, (px, py, PLACE_Z), op, f"let go of the {name}", yaw),
        Waypoint(1.0, (px, py, hover), op, "straight up", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: the specimen. Grip the board across its width at the middle, lift it, turn it a quarter turn
    anticlockwise in place (roots from +y to -x), carry it over the sheet, lay it down, let go, go up and aside."""
    sx, sy, _ = L.START["specimen"]
    op, gr = FINGERS["specimen"]
    plan = _pick("specimen", SPEC_HOVER)
    plan += [
        Waypoint(2.0, (sx, sy, SPEC_HOVER), gr, "turn it a quarter turn, roots toward the foot", YAW["specimen"][1]),
        Waypoint(0.5, (sx, sy, SPEC_HOVER), gr, "settle", YAW["specimen"][1]),
    ]
    plan += _place("specimen", SPEC_HOVER, carry=2.5)
    plan.append(Waypoint(1.5, PARK["left"], op, "out of the way", YAW["specimen"][1]))
    return plan


def _until(plan: list[Waypoint], label: str) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label == label:
            return t
    raise KeyError(label)


def _right() -> list[Waypoint]:
    """Right arm: the slip, then the label. Pick the slip and hold it up over its start until the left arm has laid
    the specimen and gone up; lay the slip 6 mm above where the label goes; then fetch the label and lay it in the
    corner (fingers on its two ends, clear of the slip); go up and aside."""
    plan = _pick("slip", HOVER)
    left = _left()
    clear = _until(left, "straight up") + 0.2  # the left gripper is up over the sheet
    wait = max(0.5, clear - sum(w.duration for w in plan))
    sx, sy, _ = L.START["slip"]
    plan.append(Waypoint(wait, (sx, sy, HOVER), FINGERS["slip"][1], "hold the slip until the left arm is up",
                         YAW["slip"][0]))
    plan += _place("slip", HOVER)
    plan += _pick("label", HOVER)
    plan += _place("label", HOVER)
    plan.append(Waypoint(1.5, PARK["right"], FINGERS["label"][0], "out of the way", YAW["label"][1]))
    return plan


def demo():
    """Both arms: the left arm lays the specimen while the right arm picks the slip and waits; then the right arm lays
    the slip and the label. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
