"""On the station table, a location-recording set-up for a choir: on the midline a grey stereo-bar plate (80 x 280 x
8 mm, centred at x 30 cm), beyond it a dark-blue choir riser (x 47-50 cm, 300 mm wide, 40 mm tall) with six singer
figures and a white mark at its centre, and in front of each arm a small pencil-condenser microphone (70 x 24 x
24 mm, 80 g, matte black with a silver grille end, the grille being the capsule end) lying in a grey foam case pad.
Rule (ORTF stereo pair): both mics flat on the bar plate, capsule points 170 mm apart, axes 110 degrees apart and
splayed outward, the pair centred on the source axis and its centre line aimed at the choir's centre mark; then let
go of both. Ordered tiers: 0.33 angle, splay and facing the choir; 0.67 and the spacing; 1.0 and centred and aimed.

Each mic is a loose mechkit compound (mechkit.py, next to this file): one rigid body of a black body box and a silver
grille box, so its capsule end shows and the scorer can read its pose. The scorer also reads the arms (TCP, pad-tip
point, opening) through rigstate.py, which wraps the embodiment's step and reset from this file's import; neither
changes anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so each mic is built as a compound body)
import rigstate  # noqa: E402
from scoring import Tracker, capsule, heading_deg, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Every tier already needs both mics still for 1 s and let go; success then holds 1 s more, so the trial cannot end
# while a gripper is still on a mic.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Rig the two small cardioid microphones on the stereo bar as an ORTF pair for the choir on the riser. The silver "
    "grille end of each mic is its capsule. Lay both mics flat on the grey bar plate, set the way an ORTF pair is "
    "set, aimed at the centre of the choir (the white mark on the riser), then let go of both mics."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


PLATE = _span_box("bar_plate", L.PLATE_X0, L.PLATE_X1, L.PLATE_Y0, L.PLATE_Y1, 0.0, L.PLATE_H, L.PLATE_GREY)
RISER = _span_box("choir_riser", L.RISER_X0, L.RISER_X1, L.RISER_Y0, L.RISER_Y1, 0.0, L.RISER_H, L.RISER_BLUE)
MARK = Cylinder("choir_mark", pos=(L.MARK[0], L.MARK[1], L.TOP + L.RISER_H + L.MARK_T / 2), radius=L.MARK_RADIUS,
                height=L.MARK_T, color=L.MARK_WHITE)
FIGURES = [
    Cylinder(f"singer_{i}", pos=(L.MARK[0], y, L.TOP + L.RISER_H + L.FIGURE_H / 2), radius=L.FIGURE_RADIUS,
             height=L.FIGURE_H, color=L.FIGURE_MAROON)
    for i, y in enumerate(L.FIGURE_Y, start=1)
]
PADS = [
    Box(f"case_pad_{n[-1]}", pos=(x, y, L.TOP + L.PAD_T / 2), size=L.PAD_SIZE, color=L.FOAM_GREY,
        yaw=math.radians(yaw))
    for n, (x, y, yaw) in L.MIC_START.items()
]


def _mic(name: str):
    x, y, yaw = L.MIC_START[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP + L.PAD_T),
        yaw=math.radians(yaw),
        mass=L.MIC_MASS,
        color=L.MIC_BLACK,
        shapes=[mk.box(L.BODY_CENTER, L.BODY_SIZE), mk.box(L.GRILLE_CENTER, L.GRILLE_SIZE, color=L.GRILLE_SILVER)],
    )


MICS = [_mic(n) for n in L.MICS]
OBJECTS = [PLATE, RISER, MARK, *FIGURES, *PADS, *MICS]

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


def _mics(objects) -> dict:
    return {n: objects[n] for n in L.MICS}


def _pose(s) -> dict:
    """A mic's capsule point (mm), heading and tilt (deg) and speed (mm/s), for the trace."""
    try:
        return {"capsule_mm": [round(float(v) * 1000, 1) for v in capsule(s)],
                "yaw_deg": round(heading_deg(s), 2), "tilt_deg": round(tilt_deg(s), 2),
                "speed_mm_s": round(float(sum(float(v) ** 2 for v in s.lin_vel) ** 0.5) * 1000, 1),
                "source": getattr(s, "source", None)}
    except (AttributeError, TypeError, ValueError):
        return {"capsule_mm": None}


def success(objects) -> bool:
    """Both mics let go, still for 1 s, flat on the bar plate with their centres over it; capsules 170 +- 10 mm apart,
    axes 110 +- 8 degrees apart and splayed outward, the pair's midpoint within 10 mm of y 0 and its bisector within
    5 degrees of the choir mark; held 1 s. This runs first every control step, so it also feeds the tracker (with the
    arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    mics = _mics(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(mics, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "score": result.score,
           "success": result.success, **result.details}
    row.update({f"{n}_pose": _pose(objects[n]) for n in L.MICS})
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   openings=[arms["left"].opening, arms["right"].opening])
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless both mics are let go (every gripper's finger zone 10 mm or more from each), still (centre path under
    1.5 mm in each of the last two 0.5 s windows, heading within 2 degrees over 1 s), lying flat (long axis within 10
    degrees of horizontal) with the centre over the bar plate and the lowest point within 5 mm of its top. Then 1.0
    for angle 110 +- 8, splayed outward, spacing 170 +- 10 mm, midpoint within 10 mm of y 0 and bisector within 5
    degrees of the mark; 0.67 without the last two; 0.33 for the angle, splay and a bisector within 30 degrees alone;
    else 0. Read from the state at hand (the last step)."""
    return _TRACKER.result(_mics(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.025  # 50 mm: 13 mm either side of a 24 mm mic
GRIP = -(L.MIC_W - 0.008) / 2  # 16 mm commanded across the mic's 24 mm
GRIP_UP = 0.017  # the TCP 17 mm above the mic's underside: pad tips 4 mm above it, holding 20 of its 24 mm
DROP = 0.001  # each mic is let go with its underside this far above the plate
APPROACH = 0.030  # each final descent starts this far above its end height and takes 1.5 s
HOVER = L.TOP + 0.10  # carried with the mic's underside 83 mm up
UP = L.TOP + 0.12  # after letting go: pad tips 107 mm up
PARK = {"left": (0.200, 0.250, L.TOP + 0.150), "right": (0.200, -0.250, L.TOP + 0.150)}
ARM_OF = {"left": "mic_a", "right": "mic_b"}
# Each mic is gripped at its centre across its two 24 mm side faces, the fingers closing square to its axis: with
# yaw 0 closing along world y, the gripper yaw equals the mic's heading, and the mic turns with the gripper. Mic A
# goes -20 -> +55 degrees (the left arm turns +75), mic B +20 -> -55 (the right arm turns -75). Joint 6 then keeps
# at least 25 degrees to its limit (yamkin, tested).
# Yaw trim, for the GPU run: parts carried in a gripper ended 0.6-1.7 degrees counter-clockwise of the commanded yaw
# in our earlier GPU runs (both arms), so each mic is laid 1 degree clockwise of its target. The error is the same
# sign on both mics, so it cancels in the included angle and only turns the bisector (5 degrees allowed). If the
# trace's pair.bisector_deg is more than 2 degrees from 0, change YAW_TRIM_DEG by minus that.
YAW_TRIM_DEG = -1.0
# Place lead, for the GPU run: parts carried in a gripper landed about 4 mm short of the aim, so each place point is
# set PLACE_LEAD past the target along the carry (toward the midline). Offline (no lag) the capsules then end 162 mm
# apart; landing 4 mm short they end 170 mm apart (window 160-180). If the trace's pair.separation_mm is off 170 by
# more than 4 mm, change PLACE_LEAD by half the difference (more lead if too wide).
PLACE_LEAD = 0.004


def pick_point(side: str) -> tuple[float, float]:
    x, y, _ = L.MIC_START[ARM_OF[side]]
    return x, y


def place_point(side: str) -> tuple[float, float]:
    """The TCP's place point: the mic's target centre plus PLACE_LEAD along the carry."""
    tx, ty = L.TARGET_CENTRE[ARM_OF[side]]
    sx, sy = pick_point(side)
    d = math.hypot(tx - sx, ty - sy)
    return tx + PLACE_LEAD * (tx - sx) / d, ty + PLACE_LEAD * (ty - sy) / d


def yaws(side: str) -> tuple[float, float]:
    """(pick, place) gripper yaw (rad)."""
    name = ARM_OF[side]
    return math.radians(L.MIC_START[name][2]), math.radians(L.TARGET_YAW_DEG[name] + YAW_TRIM_DEG)


def _pick(side: str) -> list[Waypoint]:
    """Down round the mic (slowly for the last 30 mm), grip it across its sides, lift it."""
    sx, sy = pick_point(side)
    yp, _ = yaws(side)
    z = L.TOP + L.PAD_T + GRIP_UP
    return [
        Waypoint(1.0, (sx, sy, z + APPROACH), OPEN, "down to just above the mic", yp),
        Waypoint(1.5, (sx, sy, z), OPEN, "slowly down round the mic", yp),
        Waypoint(0.5, (sx, sy, z), OPEN, "settle", yp),
        Waypoint(1.5, (sx, sy, z), GRIP, "grip the mic across its sides", yp),
        Waypoint(0.3, (sx, sy, z), GRIP, "settle the grip", yp),
        Waypoint(1.2, (sx, sy, HOVER), GRIP, "lift the mic", yp),
    ]


def _place(side: str) -> list[Waypoint]:
    """Carry the mic over the bar plate turning it to its heading, down (slowly), let go, up, out of the way."""
    px, py = place_point(side)
    _, yl = yaws(side)
    z = L.PLATE_TOP_W + DROP + GRIP_UP
    return [
        Waypoint(2.5, (px, py, HOVER), GRIP, "carry it over the bar plate, turning it", yl),
        Waypoint(0.5, (px, py, HOVER), GRIP, "settle over the plate", yl),
        Waypoint(1.0, (px, py, z + APPROACH), GRIP, "down to just above the plate", yl),
        Waypoint(1.5, (px, py, z), GRIP, "slowly lay the mic on the plate", yl),
        Waypoint(0.5, (px, py, z), GRIP, "settle", yl),
        Waypoint(1.5, (px, py, z), OPEN, "let go of the mic", yl),
        Waypoint(1.0, (px, py, UP), OPEN, "straight up", yl),
        Waypoint(1.5, PARK[side], OPEN, "out of the way", yl),
    ]


def _over(side: str, duration: float = 2.0) -> Waypoint:
    sx, sy = pick_point(side)
    return Waypoint(duration, (sx, sy, HOVER), OPEN, "over the mic", yaws(side)[0])


def _span(wps: list[Waypoint]) -> float:
    return sum(w.duration for w in wps)


def _left() -> list[Waypoint]:
    return [_over("left"), *_pick("left"), *_place("left")]


def _right() -> list[Waypoint]:
    """Waits over its mic, then picks it so that its carry starts as the left arm, its gripper already up from the
    bar, moves out of the way: the two grippers are never over the bar at once (at least 38 cm apart along y)."""
    first = _over("right")
    pick = _pick("right")
    wait = _span(_left()) - _left()[-1].duration - first.duration - _span(pick)
    hold = Waypoint(wait, first.pos, OPEN, "wait over the mic until the left arm has let go and gone up", first.yaw)
    return [first, hold, *pick, *_place("right")]


def demo():
    """Left arm first: it lays mic A on the plate, lets go and goes up; the right arm, waiting over mic B, then picks
    it and lays it, so the two grippers are never over the bar together. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
