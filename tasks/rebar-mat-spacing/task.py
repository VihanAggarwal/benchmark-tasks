"""On the station table: the plywood side forms of a small slab (a fixed rectangle, x 10 to 42 cm, y -23 to +23 cm
inside, boards 10 mm thick and 25 mm tall), two fixed grey bolsters (12 mm wide, 20 mm tall) running along x inside
it at y +-12 cm, and three loose rust-brown bars (16 x 16 mm, 400 mm long, 0.10 kg) lying along x on timber racks
outside the forms: two on the left arm's side at y +30 and +38 cm, one on the right arm's at y -32 cm. The bars go
across both bolsters, parallel, 100 mm apart centre to centre, every bar end at least 20 mm from the side forms (the
concrete cover), as a rebar worker lays a slab's bottom mat. Gripping from above, neither arm reaches the other
side's bars.

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
from scoring import Tracker, end_covers, heading_err_deg, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. The last bar rests on the bolsters (within 3 mm) while the gripper still holds it,
# from shortly before the slow descent ends; settling, letting go and lifting the hand take 0.5 + 1.5 + 1.0 s more.
# With 3 s the trial cannot end while the fingers are still closed on the bar.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Lay the three rust-brown bars across the two grey bolsters, parallel and 10 cm apart centre to centre, with at "
    "least 2 cm between each bar end and the wooden side forms."
)

FORMS = [
    Box(name, pos=(x, y, L.TOP + L.FORM_H / 2), size=(sx, sy, L.FORM_H), color=L.PLYWOOD)
    for name, x, y, sx, sy in L.FORMS
]
BOLSTERS = [
    Box(name, pos=(L.BOLSTER_X, y, L.TOP + L.BOLSTER_H / 2), size=(L.BOLSTER_LEN, L.BOLSTER_W, L.BOLSTER_H),
        color=L.BOLSTER_GREY)
    for name, y in zip(("bolster_left", "bolster_right"), L.BOLSTER_YS, strict=True)
]
RACKS = [
    Box(name, pos=(x, y, L.TOP + L.RACK_H / 2), size=(L.RACK_SIZE_X, length, L.RACK_H), color=L.DUNNAGE)
    for name, x, y, length in L.RACKS
]
BARS = [  # sized as they lie in the mat (along y), turned a quarter turn to lie along x on the racks
    Box(name, pos=(L.BAR_START_X, y, L.RACK_TOP + L.BAR_SIZE[2] / 2), size=L.BAR_SIZE, mass=L.BAR_MASS,
        color=L.RUST, yaw=L.BAR_START_YAW)
    for name, y in L.BARS
]
OBJECTS = [*FORMS, *BOLSTERS, *RACKS, *BARS]
BAR_NAMES = tuple(b.name for b in BARS)

_TRACKER = Tracker()


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


def _pose(s) -> dict:
    """A bar's centre (mm), tilt and heading error from y (deg), end covers (mm) and speed (mm/s) for the trace."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "heading_err_deg": round(heading_err_deg(s, "y"), 2),
                "covers_mm": [round(c * 1000, 1) for c in end_covers(s)],
                "speed_mm_s": round(float(sum(float(v) ** 2 for v in s.lin_vel) ** 0.5) * 1000, 1)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """Three bars on both bolsters along y, 100 +- 5 mm apart where they cross each bolster, every end 20 mm or
    more from the side forms, every bar still (SUCCESS_HOLD_S = 3 s). This runs first every control step, so it
    also feeds the tracker."""
    bars = [objects[n] for n in BAR_NAMES]
    result = _TRACKER.update(bars)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    row.update({n: _pose(objects[n]) for n in BAR_NAMES})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a bar ends off the table, on an end or edge, or on or across a form. Else 1.0 the full mat (three bars
    laid on both bolsters, spacing 100 +- 5 mm, end cover >= 20 mm); 0.75 spacing within 12 mm or a cover of
    10-20 mm; 0.5 two neighbouring bars right; 0.25 one bar laid with its ends covered; 0 otherwise. Read from the
    state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in BAR_NAMES]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
BAR_OPEN = -0.025  # 50 mm: 17 mm either side of a 16 mm bar
BAR_GRIP = -(L.BAR_SIZE[0] - 0.008) / 2  # 8 mm commanded across the bar's 16 mm: about 20 N per finger
# A bar is picked where it lies along x with the fingers closing along y (yaw 0), turned a quarter turn in the hand
# at its rack and set down along y with the fingers closing along x: left arm yaw -pi/2, right arm +pi/2 (the other
# signs run joint6 into its limit near the midline).
YAW_PICK, YAW_L, YAW_R = 0.0, -math.pi / 2, math.pi / 2
PAD = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = L.BAR_SIZE[2] / 2  # the TCP at the bar's mid-height: the pads (22 mm long) hold all 16 mm of its side,
#                              their tips 5 mm below its underside, beside it (the racks and bolsters are elsewhere)
DROP = 0.002  # each bar is let go with its underside 2 mm above the bolsters (the arm sags a little under it)
APPROACH = 0.030  # each final descent starts this far above the grip height and takes 1.5 s
HOVER = L.TOP + 0.10  # carried with the bar's underside 92 mm up: 56 mm over the forms, racks and laid bars
# Each arm holds its bar 7 cm from the bar's centre, toward the arm bases (x - 0.07 where it lies along x): the
# grip is clear of the racks (x 0.15 and 0.41) and, after the quarter turn, sits on the arm's own side of the mat
# (y +-0.07 when the bar is centred on y 0), 4 cm inside the bolsters, which keeps the far placements well inside
# reach (straight down on the midline at x 0.36 would be near its edge). Off centre, the 0.10 kg bar puts 0.07 N m
# on the grip, about a quarter of what the pads' friction holds (20 N per pad, friction about 1.2, about 6 mm of
# lever over the contact patch).
GRIP_OFF = 0.07

PICK_Z = L.RACK_TOP + GRIP_UP
PLACE_Z = L.BOLSTER_TOP + DROP + GRIP_UP

_BY = dict(L.BARS)  # bar name -> starting y
# Where each bar goes (the x of its centre line); every bar is centred on y 0, its ends 30 mm from the side forms.
# The right arm's bar first, farthest out; then the left arm's two, working back toward the arm bases so the left
# arm never reaches over a bar it has laid.
PLACE_TO = {"bar_right": L.MAT_XS[2], "bar_left_inner": L.MAT_XS[1], "bar_left_outer": L.MAT_XS[0]}
# Per-arm trim of the place point (x, y), for the GPU run: if the bars land off their marks by the same amount
# for one arm (the pads off the commanded point, the arm's sag under the bar), read the bars' pos_mm in TASK_TRACE
# and move that arm's place point by minus the offset. Offline there is nothing to trim.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _wait(wps: list[Waypoint], until: float, label: str) -> None:
    """Hold the last pose until ``until`` s on the shared clock (no-op if already past it)."""
    t = _span(wps)
    if until > t + 1e-9:
        w = wps[-1]
        wps.append(Waypoint(until - t, w.pos, w.finger, label, w.yaw))


def _over(name: str, move: float) -> list[Waypoint]:
    """To the hover point over a bar's grip on its rack, fingers open across it."""
    return [Waypoint(move, (L.BAR_START_X - GRIP_OFF, _BY[name], HOVER), BAR_OPEN, "over the bar", YAW_PICK)]


def _grip_and_turn(name: str, yaw: float) -> list[Waypoint]:
    """Down beside the bar, grip it at mid-height, lift it straight up, turn it a quarter turn over the rack."""
    x, y = L.BAR_START_X - GRIP_OFF, _BY[name]
    return [
        Waypoint(1.0, (x, y, PICK_Z + APPROACH), BAR_OPEN, "down to just above the bar", YAW_PICK),
        Waypoint(1.5, (x, y, PICK_Z), BAR_OPEN, "slowly down beside the bar", YAW_PICK),
        Waypoint(0.5, (x, y, PICK_Z), BAR_OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (x, y, PICK_Z), BAR_GRIP, "grip the bar", YAW_PICK),
        Waypoint(1.2, (x, y, HOVER), BAR_GRIP, "lift the bar", YAW_PICK),
        Waypoint(2.0, (x, y, HOVER), BAR_GRIP, "turn the bar a quarter turn over the rack", yaw),
    ]


def _place(name: str, arm: str, yaw: float) -> list[Waypoint]:
    """Carry the turned bar over its place, lower it slowly onto the bolsters, let go, go up."""
    s = 1.0 if arm == "left" else -1.0
    tx, ty = PLACE_TRIM[arm]
    x, y = PLACE_TO[name] + tx, s * GRIP_OFF + ty
    return [
        Waypoint(2.0, (x, y, HOVER), BAR_GRIP, "carry the bar over its place", yaw),
        Waypoint(1.0, (x, y, PLACE_Z + APPROACH), BAR_GRIP, "down to 3 cm above", yaw),
        Waypoint(1.5, (x, y, PLACE_Z), BAR_GRIP, "slowly set the bar down on the bolsters", yaw),
        Waypoint(0.5, (x, y, PLACE_Z), BAR_GRIP, "settle", yaw),
        Waypoint(1.5, (x, y, PLACE_Z), BAR_OPEN, "let go of the bar", yaw),
        Waypoint(1.0, (x, y, HOVER), BAR_OPEN, "up", yaw),
    ]


def _plans() -> dict[str, list[Waypoint]]:
    """Both arms on one clock. Each arm takes the bars on its own side; one bar is in the air over the mat at a
    time:
      right: its bar -> x 0.36, then back to its own side
      left:  waits over its inner bar, picks it as the right arm lets go -> x 0.26; its outer bar -> x 0.16"""
    right = _over("bar_right", 2.0) + _grip_and_turn("bar_right", YAW_R) + _place("bar_right", "right", YAW_R)
    t_right_up = _span(right)
    right.append(Waypoint(1.5, (L.BAR_START_X, -0.38, HOVER), BAR_OPEN, "back to its own side, out of the way",
                          YAW_R))

    left = _over("bar_left_inner", 2.0)
    pick = _grip_and_turn("bar_left_inner", YAW_L)
    # Lifted just as the right arm is up from its bar (the turn over the rack stays on the left side: the far end
    # sweeps no further than y +0.03); carried over the mat once the right arm is on its way out.
    _wait(left, t_right_up - _span(pick[:5]), "wait for the right bar to be down")
    left += pick + _place("bar_left_inner", "left", YAW_L)
    left += _over("bar_left_outer", 2.0) + _grip_and_turn("bar_left_outer", YAW_L)
    left += _place("bar_left_outer", "left", YAW_L)
    left.append(Waypoint(1.5, (L.BAR_START_X - GRIP_OFF, 0.36, HOVER), BAR_OPEN, "back to its own side", YAW_L))
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = _plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
