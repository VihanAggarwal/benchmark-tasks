"""On the station table, a plan of a new below-ground foul drain at 1:50 (1 m = 20 mm), drawn in dark blue on a
pale cream plan sheet (a fixed 1 mm sheet): the run starts at its free end (the head), runs as a 10 mm line (100 mm
pipe), widens to a 20 mm line (150 mm pipe), turns twice through 90 degrees and ends at the existing chamber at the
site boundary (a fixed grey cylinder 70 mm across with a lighter cover, a red boundary line beyond it). Four loose
chamber bases (black upright cylinders 60 mm across, 40 mm tall, 100 g) stand on the sheet off the route, two in
front of each arm. Rule (Approved Document H, H1 Section 2): access at the head of the run, at each bend and at the
change of pipe size, so the four chambers belong at the head, the size change and the two bends (not marked);
straights need none. Each point covered by an upright, let-go, settled chamber within 15 mm scores 0.25; all four
score 1.0 once every chamber has been still for 1 s. The scorer reads the arms (pad-tip points and openings)
through rigstate.py, which wraps the embodiment's step and reset from this file's import, only to check that each
chamber was let go.

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
from scoring import Tracker, bottom_gap, held, nearest_point, standing, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success (score 1.0, which already needs every chamber still for 1 s) must hold this long. A chamber counts only
# once let go (no pad-tip point lower than 10 mm above its top within the finger reach). Should the arms ever be
# unreadable, the last chamber set down would count from the end of its 0.5 s settle while still gripped; letting
# go takes 1.5 s more, so with 2 s the trial still cannot end while the fingers are closed on a chamber.
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "This plan, drawn at 1:50 (1 m = 20 mm), shows a new below-ground foul drain. It runs from its free end, the "
    "head of the run, to the existing chamber at the boundary; it is laid at one fall and has no junctions. The "
    "thin line is 100 mm pipe, the thick line 150 mm pipe. Set the four inspection chamber bases on the route where "
    "Approved Document H requires access points on this run, each one upright and centred on the line, then let "
    "go of them."
)


def _flat(name: str, rect, z0: float, h: float, color) -> Box:
    x0, x1, y0, y1 = rect
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, z0 + h / 2), size=(x1 - x0, y1 - y0, h), color=color)


SHEET = _flat("plan_sheet", L.SHEET, L.TOP, L.SHEET_H, L.SHEET_CREAM)
ROUTE = [_flat(name, (x0, x1, y0, y1), L.SHEET_TOP, L.LINE_T, L.ROUTE_BLUE) for name, x0, x1, y0, y1 in L.ROUTE]
BOUNDARY = _flat("site_boundary", (L.SHEET[0], L.SHEET[1], L.BOUNDARY_Y - L.BOUNDARY_W / 2,
                                   L.BOUNDARY_Y + L.BOUNDARY_W / 2), L.SHEET_TOP, L.LINE_T, L.BOUNDARY_RED)
EXISTING = Cylinder("existing_chamber", pos=(L.EXIST[0], L.EXIST[1], L.SHEET_TOP + L.EXIST_H / 2),
                    radius=L.EXIST_RADIUS, height=L.EXIST_H, color=L.CONCRETE)
COVER = Cylinder("existing_cover", pos=(L.EXIST[0], L.EXIST[1], L.SHEET_TOP + L.EXIST_H + L.COVER_H / 2),
                 radius=L.COVER_RADIUS, height=L.COVER_H, color=L.COVER_GREY)


def _chamber(name: str) -> Cylinder:
    x, y = L.START[name]
    return Cylinder(name, pos=(x, y, L.SHEET_TOP + L.CHAMBER_H / 2), radius=L.CHAMBER_RADIUS, height=L.CHAMBER_H,
                    mass=L.CHAMBER_MASS, color=L.CHAMBER_BLACK)


CHAMBERS = [_chamber(n) for n in L.NAMES]
OBJECTS = [SHEET, *ROUTE, BOUNDARY, EXISTING, COVER, *CHAMBERS]
CHAMBER_NAMES = tuple(c.name for c in CHAMBERS)

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the chambers teleporting back)."""
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


def _pose(s, path, arms) -> dict:
    """A chamber's centre (mm), tilt (deg), bottom above the sheet top (mm), the nearest required point and the
    centre's offset from it (mm, x and y and planar), its path over the last 0.5 s (mm), whether a gripper still
    holds it and whether it stands on the plan, for the trace."""
    try:
        point, dist = nearest_point(s)
        px, py = L.POINTS[point]
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "bottom_mm": round(bottom_gap(s) * 1000, 1),
                "nearest": point,
                "off_mm": round(dist * 1000, 1),
                "off_xy_mm": [round((float(s.pos[0]) - px) * 1000, 1), round((float(s.pos[1]) - py) * 1000, 1)],
                "path_mm": None if path is None else round(path * 1000, 2),
                "held": held(s, arms),
                "standing": standing(s)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """1.0 now: all four points (head, size change, both bends) covered, each by an upright chamber on the sheet,
    let go, its centre within 15 mm of the point and moving under 2 mm over the last 0.5 s, and every chamber still
    for 1 s (under 2 mm in each of the last two 0.5 s windows); held SUCCESS_HOLD_S = 2 s. This runs first every
    control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    chambers = [objects[n] for n in CHAMBER_NAMES]
    result = _TRACKER.update(chambers, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if isinstance(arms, dict):  # each arm on its own: a missing one is traced as None, never a crash
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    paths = _TRACKER.paths() if result.details.get("valid_state") else None
    row.update({n: _pose(objects[n], None if paths is None else paths[i], arms)
                for i, n in enumerate(CHAMBER_NAMES)})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each required point (head, size change, bend 1, bend 2) covered by a counted chamber: upright within
    10 degrees, lowest point within 3 mm of the sheet top, let go (no gripper's pad-tip point lower than 10 mm above
    its top within that gripper's finger reach, 95 mm fully open, 80 mm closed on a chamber), centre path under 2 mm
    over the last 0.5 s, centre within 15 mm of the point. All four covered: 1.0 once every chamber has been still
    for 1 s, 0.75 until then. Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in CHAMBER_NAMES], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.040  # 80 mm: 10 mm either side of a chamber
GRIP = -(2 * L.CHAMBER_RADIUS - 0.008) / 2  # 52 mm commanded across the 60 mm chamber
YAW = 0.0  # both arms: the fingers close along y (a round chamber can be gripped at any heading)
GRIP_UP = 0.024  # the TCP this far above the chamber's bottom: the pad tips 11 mm above it, the pads on 11-40 mm
DROP = 0.002  # each chamber is let go with its bottom 2 mm above the sheet (1.5 mm above a drawn line)
PICK_Z = L.SHEET_TOP + GRIP_UP
PLACE_Z = L.SHEET_TOP + DROP + GRIP_UP
SLOW = 0.020  # the last 20 mm down onto a chamber, or onto the plan, are taken slowly
HOVER_Z = L.SHEET_TOP + 0.085  # travel: a carried chamber's bottom 61 mm up, 21 mm over a standing chamber's top
PARK = {"left": (0.280, 0.230, L.SHEET_TOP + 0.140), "right": (0.280, -0.290, L.SHEET_TOP + 0.140)}
# Parts land a few mm short of the commanded point along the way they were carried (arms trail under load): each
# place point is moved this far on, along the carry.
LAND_SHORT = 0.004
# Where each arm takes which chamber, in order (both arms on one clock):
#   1. right: chamber_1 to the head H; left: chamber_3 to the first bend B1 (same x, 310 mm apart);
#   2. right: chamber_2 to the size change P, then it parks; left: chamber_4 to the second bend B2, but only after
#      waiting WAIT_BEFORE s open over chamber_4, so it comes down at B2 once the right arm has left P (with both
#      at once their elbows would pass 38 mm apart).
# Then each arm parks over its own side, clear of the route.
PLAN = {
    "right": (("chamber_1", "head"), ("chamber_2", "size_change")),
    "left": (("chamber_3", "bend_1"), ("chamber_4", "bend_2")),
}
# Per-arm trim of the place points (x, y), for the GPU run: if one arm's chambers land off their points by the same
# amount, read the chambers' off_xy_mm in TASK_TRACE and move that arm's place points by minus the offset.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
WAIT_BEFORE = {"left": (0.0, 7.0), "right": (0.0, 0.0)}  # s spent over each chamber, open, before going down to it


def place_point(name: str, point: str, arm: str) -> tuple[float, float]:
    """The commanded place point: the access point, LAND_SHORT further along the carry, plus the arm's trim."""
    x0, y0 = L.START[name]
    px, py = L.POINTS[point]
    d = math.hypot(px - x0, py - y0)
    tx, ty = PLACE_TRIM[arm]
    return px + LAND_SHORT * (px - x0) / d + tx, py + LAND_SHORT * (py - y0) / d + ty


def _move(name: str, point: str, arm: str, k: int) -> list[Waypoint]:
    """Over a chamber (waiting there if the plan says so), slowly down round it, grip it, lift it, carry it over its
    point, set it down slowly on the plan, let go, straight up."""
    x, y = L.START[name]
    px, py = place_point(name, point, arm)
    wait = WAIT_BEFORE[arm][k]
    return [
        Waypoint(2.5 if k == 0 else 1.8, (x, y, HOVER_Z), OPEN, f"over {name}", YAW),
        *([Waypoint(wait, (x, y, HOVER_Z), OPEN, "wait for the other arm", YAW)] if wait > 0 else []),
        Waypoint(1.0, (x, y, PICK_Z + SLOW), OPEN, "down to just above it", YAW),
        Waypoint(1.2, (x, y, PICK_Z), OPEN, "slowly down round it", YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", YAW),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", YAW),
        Waypoint(2.0, (px, py, HOVER_Z), GRIP, f"carry it over the {point.replace('_', ' ')}", YAW),
        Waypoint(1.0, (px, py, PLACE_Z + SLOW), GRIP, "down to just above the plan", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly stand it on the plan", YAW),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", YAW),
        Waypoint(1.0, (px, py, HOVER_Z), OPEN, "straight up, clear of it", YAW),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for k, (name, point) in enumerate(PLAN[arm]):
        wps += _move(name, point, arm, k)
    return wps + [Waypoint(1.5, PARK[arm], OPEN, "park, clear of the route", YAW)]


def demo():
    """Both arms on one clock, each setting its own side's two chambers. DEMO_HOLD_ONLY=1: no plan (both arms hold
    the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
