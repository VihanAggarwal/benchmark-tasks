"""On the station table: a green circuit board (150 x 220 x 4 mm, fixed) on the midline 30 cm in front of the arm
bases, carrying three surface-mount footprints at 8:1 scale, each a pair of tinned pads along x: a 2512 resistor's
at y +6.5 cm, a 1210 capacitor's at y 0 and a D-case tantalum capacitor's at y -6.5 cm, with a white + mark beside the
tantalum footprint's far pad. On a grey antistatic mat on the left lie the resistor (black, silver ends) and the
ceramic capacitor (tan, silver ends); on a mat on the right lies the tantalum capacitor (amber, a dark band across its
+ end). Rule: each part goes on its own pads, square, with no end past a pad's outer end and its sides sticking out
past the pads by at most 25 % of its width (IPC-A-610 Class 3; 50 %: Classes 1-2), the tantalum's band on the + pad.
Flat means on a face: the resistor either broad face up, the ceramic capacitor (square in section) on any long
face, the tantalum band face up.

The parts are loose mechkit compounds (mechkit.py, next to this file): each one rigid body made of several boxes, so
the terminations and the band are visible and the scorer can read each part's pose (base frame and orientation).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the right arm places the tantalum, the left arm the resistor and the ceramic capacitor (straight down, each
arm reaches only the parts on its own side; leaned about 20 degrees either reaches the other mat); with one arm idle
the scripted demo scores what the other arm's own parts earn (left alone 0.67, right alone 0.33).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compound bodies)
from scoring import Tracker, centre_offset, rotation  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The last part is lowered to 1 mm above its pads and counts as lying on them
# (lowest point within 2 mm) while the gripper still holds it; success then needs 0.5 s at rest. Settling, letting go
# and going up take 0.5 + 1.5 + 1.0 s, so with 2.5 s the trial ends with the fingers open and the hand going up.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Place the three parts on the circuit board, each on the pair of silver pads that matches its size (pads a "
    "little wider than the part and reaching just past both of its ends): square to its pads and centred on them, "
    "with neither end sticking out past a pad. The tantalum capacitor, the largest part, goes with its dark band on "
    "the + pad, the one with the small white square (the + mark) beside it."
)


def _pad(name: str, x: float, y: float, fp: L.Footprint) -> Box:
    return Box(name, pos=(x, y, L.TOP + L.BOARD_TOP + L.PAD_T / 2), size=(fp.pad_len, fp.pad_width, L.PAD_T),
               color=L.TIN)


BOARD = Box("board", pos=(L.BOARD_X, L.BOARD_Y, L.TOP + L.BOARD_SIZE[2] / 2), size=L.BOARD_SIZE, color=L.PCB_GREEN)
(_rn, _rnx, _rny), (_rf, _rfx, _rfy) = L.R_FP.pads()
(_cn, _cnx, _cny), (_cf, _cfx, _cfy) = L.C_FP.pads()
(_tn, _tnx, _tny), (_tf, _tfx, _tfy) = L.T_FP.pads()
PADS = [
    _pad(_rn, _rnx, _rny, L.R_FP), _pad(_rf, _rfx, _rfy, L.R_FP),
    _pad(_cn, _cnx, _cny, L.C_FP), _pad(_cf, _cfx, _cfy, L.C_FP),
    _pad("t_pad_minus", _tnx, _tny, L.T_FP), _pad("t_pad_plus", _tfx, _tfy, L.T_FP),
]
PLUS_MARK = Box("plus_mark", pos=(L.PLUS_MARK_X, L.PLUS_MARK_Y, L.TOP + L.BOARD_TOP + L.PLUS_MARK_SIZE[2] / 2),
                size=L.PLUS_MARK_SIZE, color=L.MARK_WHITE)
MATS = [Box(name, pos=(x, y, L.TOP + L.MAT_TOP / 2), size=L.MAT_SIZE, color=L.MAT_GREY)
        for name, (x, y) in (("mat_left", L.MAT_L), ("mat_right", L.MAT_R))]


def _chip(part: L.Part, body_color) -> mk.Assembly:
    """A chip part with wrap-around terminations: a body box between two silver end boxes, full height."""
    h, t = part.height, part.term
    return mk.assembly(
        part.name,
        origin=(part.start[0], part.start[1], L.TOP + L.MAT_TOP),
        yaw=part.start_yaw,
        mass=part.mass,
        color=L.TIN,
        shapes=[
            mk.box((0.0, 0.0, h / 2), (part.length - 2 * t, part.width, h), color=body_color),
            mk.box((part.length / 2 - t / 2, 0.0, h / 2), (t, part.width, h)),
            mk.box((-(part.length / 2 - t / 2), 0.0, h / 2), (t, part.width, h)),
        ],
    )


def _tantalum(part: L.Part) -> mk.Assembly:
    """The moulded tantalum: an amber body on two bottom-only silver termination plates, with a dark band across
    its top at the + (local +x) end."""
    t, w = part.term, part.width
    body_z = L.TA_PLATE_T + L.TA_BODY_H / 2
    return mk.assembly(
        part.name,
        origin=(part.start[0], part.start[1], L.TOP + L.MAT_TOP),
        yaw=part.start_yaw,
        mass=part.mass,
        color=L.TIN,
        shapes=[
            mk.box((0.0, 0.0, body_z), (part.length, w, L.TA_BODY_H), color=L.TA_AMBER),
            mk.box((part.length / 2 - t / 2, 0.0, L.TA_PLATE_T / 2), (t, w, L.TA_PLATE_T)),
            mk.box((-(part.length / 2 - t / 2), 0.0, L.TA_PLATE_T / 2), (t, w, L.TA_PLATE_T)),
            mk.box((part.length / 2 - L.TA_BAND_W / 2, 0.0, L.TA_PLATE_T + L.TA_BODY_H + L.TA_BAND_T / 2),
                   (L.TA_BAND_W, w, L.TA_BAND_T), color=L.TA_BAND),
        ],
    )


RESISTOR = _chip(L.RESISTOR, L.R_BLACK)
CAPACITOR = _chip(L.CAPACITOR, L.C_TAN)
TANTALUM = _tantalum(L.TANTALUM)
OBJECTS = [BOARD, *PADS, PLUS_MARK, *MATS, RESISTOR, CAPACITOR, TANTALUM]

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
    """All three parts at rest on their own pads to Class 3 (no end overhang, side overhang at most 25 % of the
    part's width), the tantalum's band on the + pad, nothing off the table; held 2.5 s. This runs first every control
    step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    result = _TRACKER.update(objects)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        for p in L.PARTS:
            st = objects[p.name]
            row[f"{p.name}_origin"] = st.origin
            row[f"{p.name}_yaw_deg"] = round(math.degrees(math.atan2(rotation(st.quat_xyzw)[1, 0],
                                                                     rotation(st.quat_xyzw)[0, 0])), 2)
            row[f"{p.name}_offset_mm"] = [round(v * 1000, 2) for v in centre_offset(p, st)]
            row[f"{p.name}_source"] = getattr(st, "source", None)
    except (AttributeError, TypeError, ValueError, KeyError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any part is off the table or the tantalum lies on its own pads reversed (band on the - pad). Otherwise
    the sum over the three parts, each counted only at rest on its own pads (flat, square within 10 degrees, no end
    past a pad's outer end, the tantalum band on +): 1/3 if its side overhang is at most 25 % of its width (Class 3),
    1/6 if at most 50 % (Classes 1-2), else 0. Read from the state at hand (the last step)."""
    return _TRACKER.result(objects).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap. Each part is gripped across its width with the
# fingers commanded 8 mm narrower, and let go by opening only to its width plus 16 mm (8 mm clear each side), so the
# open fingers stay clear of the neighbouring footprints 65 mm away.
OPEN = -0.04695


def grip(part: L.Part) -> float:
    return -(part.width - 0.008) / 2


def release(part: L.Part) -> float:
    return -(part.width + 0.016) / 2


# Grip heights: the TCP this far above the part's underside (pad tips 13 mm below the TCP). Resistor and ceramic
# capacitor (20 mm tall): pad tips 7 mm above the underside, 13 mm held. Tantalum (24 mm): 9 mm up, 15 mm held, clear
# of the band at its end.
GRIP_UP = {"resistor": 0.020, "capacitor": 0.020, "tantalum": 0.022}
DROP = 0.001  # each part is let go with its underside 1 mm above its pads
APPROACH = 0.030  # each final descent starts this far above the grip height and takes 1.5 s (pick) or 2 s (place)
HOVER = L.TOP + 0.10  # carried with the parts' undersides 75-80 mm above the table
# Picked at yaw -+90 degrees (fingers closing along x, across the part's width, which lies along x on the mat) and set
# down at yaw 0 (fingers along y): the quarter turn while carrying lays each part's long side along x. Near the midline
# the left arm needs yaw -90 and the right arm +90 (the other sign runs joint6 toward its limit); the right arm's turn
# from +90 to 0 swings the tantalum's band from +y to +x, onto the + pad.
YAW = {"left": -math.pi / 2, "right": math.pi / 2}
PLACE_YAW = 0.0
# Trims of the place points (x, y), for the GPU run: if a part lands off its footprint's centre (<part>_offset_mm in
# TASK_TRACE), move its place point by minus the offset. The parts weigh 20-50 g, so the arms' sag is about 1 mm.
PLACE_TRIM = {"resistor": (0.0, 0.0), "capacitor": (0.0, 0.0), "tantalum": (0.0, 0.0)}
PARK = {"left": (0.20, 0.22, HOVER), "right": (0.20, -0.25, HOVER)}
# The two grippers never work over the board at the same time (their footprints are only 6.5-13 cm apart): the right
# arm places the tantalum first and goes aside; the left arm, which meanwhile comes over the resistor and waits, then
# places the resistor and the ceramic capacitor.
LEFT_WAIT_S = 9.5


def pick_point(part: L.Part) -> tuple[float, float, float]:
    return (part.start[0], part.start[1], L.TOP + L.MAT_TOP + GRIP_UP[part.name])


def place_point(part: L.Part) -> tuple[float, float, float]:
    dx, dy = PLACE_TRIM[part.name]
    return (part.footprint.x + dx, part.footprint.y + dy, L.PAD_TOP_W + DROP + GRIP_UP[part.name])


def _pick(part: L.Part, arm: str, over: float = 2.0) -> list[Waypoint]:
    """Come over the part, down beside it, grip it across its width and lift it."""
    px, py, pz = pick_point(part)
    yaw, g = YAW[arm], grip(part)
    return [
        Waypoint(over, (px, py, HOVER), OPEN, f"over the {part.name}", yaw),
        Waypoint(1.0, (px, py, pz + APPROACH), OPEN, f"down to just above the {part.name}", yaw),
        Waypoint(1.5, (px, py, pz), OPEN, f"slowly down beside the {part.name}", yaw),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", yaw),
        Waypoint(1.5, (px, py, pz), g, f"grip the {part.name} across its width", yaw),
        Waypoint(1.5, (px, py, HOVER), g, f"lift the {part.name}", yaw),
    ]


def _place(part: L.Part) -> list[Waypoint]:
    """Carry the part over its footprint turning it a quarter turn, lower it slowly onto its pads, let go, go up."""
    qx, qy, qz = place_point(part)
    g, r = grip(part), release(part)
    return [
        Waypoint(2.5, (qx, qy, HOVER), g, f"carry the {part.name} over its pads, turning it a quarter turn",
                 PLACE_YAW),
        Waypoint(0.8, (qx, qy, HOVER), g, "settle over the pads", PLACE_YAW),
        Waypoint(1.0, (qx, qy, qz + APPROACH), g, "down to 3 cm above the pads", PLACE_YAW),
        Waypoint(2.0, (qx, qy, qz), g, f"slowly lower the {part.name} onto its pads", PLACE_YAW),
        Waypoint(0.5, (qx, qy, qz), g, "settle", PLACE_YAW),
        Waypoint(1.5, (qx, qy, qz), r, f"let go of the {part.name}", PLACE_YAW),
        Waypoint(1.0, (qx, qy, HOVER), r, "straight up", PLACE_YAW),
    ]


def _right() -> list[Waypoint]:
    """Right arm: the tantalum capacitor onto its footprint, band on the + pad; then aside."""
    return [*_pick(L.TANTALUM, "right"), *_place(L.TANTALUM),
            Waypoint(1.5, PARK["right"], OPEN, "aside", PLACE_YAW)]


def _left() -> list[Waypoint]:
    """Left arm: over the resistor, wait for the right arm to finish at the board, then the resistor and the
    ceramic capacitor onto their footprints; then aside."""
    rx, ry, _ = pick_point(L.RESISTOR)
    first = _pick(L.RESISTOR, "left")
    wait = Waypoint(LEFT_WAIT_S, (rx, ry, HOVER), OPEN, "wait over the resistor while the right arm works",
                    YAW["left"])
    return [first[0], wait, *first[1:], *_place(L.RESISTOR),
            *_pick(L.CAPACITOR, "left"), *_place(L.CAPACITOR),
            Waypoint(1.5, PARK["left"], OPEN, "aside", PLACE_YAW)]


def demo():
    """Both arms, taking turns at the board. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling
    check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
