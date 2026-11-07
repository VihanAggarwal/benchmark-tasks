"""On the station table: a light grey site sheet (380 x 620 x 1 mm, fixed) spans x 0.080 to 0.460, y -0.310 to
+0.310, standing for the ground of a site under investigation. Three soil stains lie on it, each built from 1 mm
vinyl sheets in three tones: a pale tan halo of two abutting lobes, and on it a mid-brown streak and a near-black
50 x 50 mm core that abut, the core set off-centre at one end of the streak (a drip point and its run-off). Four
orange pin flags (cylinders 25 mm across, 60 mm tall, 27 g) stand on the bare table beside the sheet, two on each
side. Judgmental (biased) sampling design: one flag on the darkest, most impacted spot of each stain (its core, not
its middle), and one background flag wholly on clean sheet at least 150 mm from every stain's outline, pale halo
included. Score 0.25 per sampling point with a planted flag on it.

The scorer reads the arms (pad-tip points and openings) through rigstate.py, which wraps the embodiment's step and reset from
this file's import; it changes nothing the embodiment does. It is used only to check that each flag was let go.

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
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Mark the sampling points on the grey site sheet: stand one orange flag upright on the darkest spot of each "
    "stain, and one flag wholly on clean grey sheet at least 15 cm from the edge of every stain, pale halo "
    "included. Then let go."
)


def _sheet(name: str, rect, layer: int, color) -> Box:
    """A fixed 1 mm vinyl sheet whose top is ``layer`` mm above the table."""
    cx, cy, sx, sy = rect
    return Box(name, pos=(cx, cy, L.TOP + L.LAYER * (layer - 0.5)), size=(sx, sy, L.LAYER), color=color)


SITE_SHEET = _sheet("site_sheet", L.SHEET, 1, L.SHEET_GREY)
STAIN_SHEETS = []
for _stain, _parts in L.STAINS.items():
    _s = _stain.lower()
    STAIN_SHEETS += [_sheet(f"{_s}_halo_{i}", lobe, 2, L.HALO_TAN) for i, lobe in enumerate(_parts["halo"])]
    STAIN_SHEETS.append(_sheet(f"{_s}_streak", _parts["streak"], 3, L.STREAK_BROWN))
    STAIN_SHEETS.append(_sheet(f"{_s}_core", _parts["core"], 3, L.CORE_BLACK))
FLAGS = [
    Cylinder(name, pos=(x, y, L.TOP + L.FLAG_HEIGHT / 2), radius=L.FLAG_RADIUS, height=L.FLAG_HEIGHT,
             mass=L.FLAG_MASS, color=L.FLAG_ORANGE)
    for name, x, y in L.FLAG_STARTS
]
OBJECTS = [SITE_SHEET, *STAIN_SHEETS, *FLAGS]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the flags teleporting back)."""
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


def _flags(objects) -> dict:
    return {name: objects[name] for name in L.FLAG_NAMES}


def success(objects) -> bool:
    """All four sampling points marked: a planted flag (upright within 10 deg, lowest point within 5 mm of the
    table top, base centre's path under 2 mm over 0.5 s, let go and clear: no gripper's pad-tip point lower than
    10 mm above its top with its axis within that gripper's finger reach, 78 mm fully open, 48 mm closed on a flag)
    with its base centre in each stain's 40 x 40 mm core target, and one at least 150 mm from every stain on the
    sheet (15 mm inside its edges). Held for 1 s. This runs first every control step, so it also feeds the tracker
    (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    result = _TRACKER.update(_flags(objects), arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
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
    """0.25 for each of the four sampling points (the three stains' cores and the background) with a planted flag
    on it: 0, 0.25, 0.5, 0.75 or 1.0."""
    return _TRACKER.result(_flags(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
# Yaw 0 throughout: the fingers close along y.
OPEN = -0.025  # 50 mm: 12.5 mm clear of the 25 mm flag on each side
FLAG_GRIP = -(2 * L.FLAG_RADIUS - 0.008) / 2  # 17 mm commanded on the 25 mm flag
GRIP_UP = 0.040  # the TCP 40 mm above the flag's bottom: pad tips 27 mm up, the pads on the flag's top 33 mm
TRAVEL_Z = L.TOP + 0.100  # not 0.12: the right arm keeps more joint-4 margin over the S2 core near the midline
# CLEARANCE: at TRAVEL_Z a carried flag's bottom is 60 mm up (100 - 40), 1 mm below the top of a flag standing on
# the sheet (61 mm) and 3 mm below one standing on a core (63 mm). So no carry path may cross a placed flag. The
# demo's order keeps every carry clear (left: far flag to S1, then near flag to the background; right: far flag to
# S2, then near flag to S3; test_carry_paths_cross_no_placed_flag). Reordering the demo needs a higher TRAVEL_Z or
# a check of every carry path against the flags already standing.
PICK_Z = L.TOP + GRIP_UP
SLOW = 0.020  # the last 20 mm of each set-down are lowered slowly
# The 27 g flag hardly loads the arm (well under 1 mm of sag), so no place trim is needed. On the GPU read each
# flag's base_x_mm, base_y_mm, bottom_mm, tilt_deg and path_mm in TASK_TRACE.


def place_z(surface_top: float) -> float:
    """TCP height that lets a flag go with its bottom 1 mm above a surface."""
    return surface_top + 0.001 + GRIP_UP


def _start(name: str) -> tuple[float, float]:
    return next((x, y) for n, x, y in L.FLAG_STARTS if n == name)


def _core(stain: str) -> tuple[float, float]:
    cx, cy, _, _ = L.STAINS[stain]["core"]
    return cx, cy


def _move(flag: str, place_xy, surface_top: float, what: str) -> list[Waypoint]:
    """Pick one flag straight down, carry it at 10 cm, stand it on its sampling point, let go, go up: 13.2 s."""
    (fx, fy), (px, py) = _start(flag), place_xy
    pz = place_z(surface_top)
    return [
        Waypoint(1.8, (fx, fy, TRAVEL_Z), OPEN, f"over {flag}"),
        Waypoint(1.2, (fx, fy, PICK_Z), OPEN, f"down round {flag}"),
        Waypoint(0.4, (fx, fy, PICK_Z), OPEN, "settle"),
        Waypoint(1.5, (fx, fy, PICK_Z), FLAG_GRIP, f"grip {flag}"),
        Waypoint(1.2, (fx, fy, TRAVEL_Z), FLAG_GRIP, "lift"),
        Waypoint(2.2, (px, py, TRAVEL_Z), FLAG_GRIP, f"carry it over {what}"),
        Waypoint(0.8, (px, py, pz + SLOW), FLAG_GRIP, "down to 2 cm above"),
        Waypoint(1.2, (px, py, pz), FLAG_GRIP, "lower the last 2 cm slowly"),
        Waypoint(0.4, (px, py, pz), FLAG_GRIP, "settle"),
        Waypoint(1.5, (px, py, pz), OPEN, "let go"),
        Waypoint(1.0, (px, py, TRAVEL_Z), OPEN, "straight up, clear of the flag"),
    ]


def _left() -> list[Waypoint]:
    """Left arm: its far flag to the S1 core, then its near flag to the background point; it then holds 10 cm
    above the background flag."""
    return (_move("flag_l2", _core("S1"), L.CORE_TOP, "the S1 core")
            + _move("flag_l1", L.BACKGROUND_POINT, L.SHEET_TOP, "the background point"))


def _right() -> list[Waypoint]:
    """Right arm, at the same time: its far flag to the S2 core, then its near flag to the S3 core; it then holds
    10 cm above the S3 flag."""
    return (_move("flag_r2", _core("S2"), L.CORE_TOP, "the S2 core")
            + _move("flag_r1", _core("S3"), L.CORE_TOP, "the S3 core"))


def demo():
    """Both arms on one clock, each with the two flags on its own side (this order keeps the arms about 8 cm
    apart; marking the background first would bring them within 5 cm). No final retreat: it would push the run
    past 600 steps. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
