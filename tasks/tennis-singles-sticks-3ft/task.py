"""On the station table, at 1:15: the middle of a tennis court painted for doubles, 150 mm (2.25 m) either side of
the net. A blue court with white lines (the singles and doubles sidelines and the centre service line, all stopping at
the net), a green surround outside the doubles sidelines, a dark grey band along the net line (x 0.350) where a
doubles net hangs, and the two net posts on that line 0.914 m outside the doubles court. Two loose white singles
sticks (a 20 mm square post on a 40 mm square foot, 71 mm tall, 40 g) stand on the near court, one each side of the
centre line. Rule (ITF Rules of Tennis, Rule 1): for singles with a doubles net, the net is held up by two singles
sticks whose centres are 0.914 m (3 ft) outside the singles court on each side: |y| 0.3353 m here, two thirds of the
way across each alley. The instruction does not give that offset. Set each stick upright on the net line there and
let go; 0.5 per side, 0.25 for a stick elsewhere in its alley.

The sticks are loose mechkit compounds (mechkit.py, next to this file): one rigid body of two boxes each, so the
scorer reads each stick's base frame (the middle of its foot's underside) and orientation. The scorer also reads the
arms (pad-tip points and openings) through rigstate.py, which wraps the embodiment's step and reset from this file's
import, only to check that each stick was let go; neither changes anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so each stick is built as a compound body)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. A stick counts only once let go (no pad-tip point lower than 10 mm above its top
# within the finger reach) and still for 1 s. Should the arms ever be unreadable, a stick would count from 1 s after
# its slow descent ends while still gripped; settling and letting go take 0.5 + 1.5 s, so with 2 s more the trial
# still cannot end while the fingers are on a stick.
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "This doubles court is hosting a singles match next, and the net is a doubles net (the dark band across the "
    "court marks where it hangs). The court is painted at 1:15 scale with the standard ITF lines. Put the two "
    "singles sticks in on the net line, one on each side of the court, standing upright where the rules put them, "
    "then let go."
)


def _mat(name: str, x0, x1, y0, y1, color) -> Box:
    """A fixed 1 mm mat on the table, given by its extents; every mat's top is the court surface."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.MAT_H / 2), size=(x1 - x0, y1 - y0, L.MAT_H),
               color=color)


def _line(name: str, x0, x1, y) -> Box:
    """A painted line along x centred on y, 3 mm wide and 0.4 mm proud of the court."""
    return Box(name, pos=((x0 + x1) / 2, y, L.COURT_TOP + L.LINE_T / 2), size=(x1 - x0, L.LINE_W, L.LINE_T),
               color=L.LINE_WHITE)


X0, X1 = L.COURT_X
D, S, B, P = L.DOUBLES_HALF, L.SURROUND_Y, L.NET_BAND_HALF, L.NET_X0  # shorthand for the mats' edges
MATS = [
    # the net band, flush with the court, post centre to post centre
    _mat("net_band", L.NET_X0, L.NET_X1, -B, B, L.NET_GREY),
    # the blue court (doubles court) either side of the band
    _mat("court_near", X0, L.NET_X0, -D, D, L.COURT_BLUE),
    _mat("court_far", L.NET_X1, X1, -D, D, L.COURT_BLUE),
    # the green surround outside the doubles sidelines, either side of the band, and beyond the band's ends
    _mat("surround_left_near", X0, L.NET_X0, D, S, L.SURROUND_GREEN),
    _mat("surround_left_far", L.NET_X1, X1, D, S, L.SURROUND_GREEN),
    _mat("surround_left_end", L.NET_X0, L.NET_X1, B, S, L.SURROUND_GREEN),
    _mat("surround_right_near", X0, L.NET_X0, -S, -D, L.SURROUND_GREEN),
    _mat("surround_right_far", L.NET_X1, X1, -S, -D, L.SURROUND_GREEN),
    _mat("surround_right_end", L.NET_X0, L.NET_X1, -S, -B, L.SURROUND_GREEN),
]
LINES = []
for _half, (_a, _b) in (("near", (X0, L.NET_X0)), ("far", (L.NET_X1, X1))):
    LINES += [
        _line(f"singles_line_left_{_half}", _a, _b, L.SINGLES_LINE_Y),
        _line(f"singles_line_right_{_half}", _a, _b, -L.SINGLES_LINE_Y),
        _line(f"doubles_line_left_{_half}", _a, _b, L.DOUBLES_LINE_Y),
        _line(f"doubles_line_right_{_half}", _a, _b, -L.DOUBLES_LINE_Y),
        _line(f"centre_line_{_half}", _a, _b, L.CENTRE_LINE_Y),
    ]
NET_POSTS = [
    Cylinder(f"net_post_{side}", pos=(L.NET_X, sg * L.NET_POST_Y, L.COURT_TOP + L.NET_POST_H / 2),
             radius=L.NET_POST_RADIUS, height=L.NET_POST_H, color=L.POST_GREEN)
    for side, sg in (("left", 1.0), ("right", -1.0))
]


def _stick(name: str) -> mk.Assembly:
    x, y = L.START[name]
    return mk.assembly(name, origin=(x, y, L.COURT_TOP), yaw=L.START_YAW, mass=L.STICK_MASS, color=L.STICK_WHITE,
                       shapes=[mk.box(L.FOOT_CENTER, L.FOOT_SIZE), mk.box(L.POST_CENTER, L.POST_SIZE)])


STICKS = [_stick(n) for n in L.NAMES]
STICK_NAMES = tuple(s.name for s in STICKS)
OBJECTS = [*MATS, *LINES, *NET_POSTS, *STICKS]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and both sticks teleporting back to
    their marks are two more signals)."""
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


def success(objects) -> bool:
    """1.0 now: on each side of the court a stick let go, still for 1 s (its axis midpoint's path under 1.5 mm in each of
    the last two 0.5 s windows), upright within 10 degrees with its lowest point within 3 mm of the court, its centre
    within 6 mm of the net line and within 8 mm of |y| 0.3353 m; held SUCCESS_HOLD_S = 2 s. This runs first every
    control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    sticks = [objects[n] for n in STICK_NAMES]
    result = _TRACKER.update(sticks, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    if isinstance(arms, dict):  # each arm on its own: a missing one is traced as None, never a crash
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    for n, s in zip(STICK_NAMES, sticks, strict=True):
        try:
            row[f"{n}_origin"] = s.origin
            row[f"{n}_quat"] = s.quat_xyzw
            row[f"{n}_source"] = getattr(s, "source", None)
        except AttributeError:
            row[f"{n}_state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """Per side of the court (y > 0, y < 0), the best of its sticks: 0 unless a stick is let go (no gripper's pad-tip
    point lower than 10 mm above its top within that gripper's finger reach), still for 1 s, upright within 10
    degrees, standing on the court (lowest point within 3 mm of it) and on the net line (centre within 6 mm of it);
    then 0.5 with its centre within 8 mm of |y| 0.3353 m (0.914 m outside the singles court at 1:15), else 0.25 with
    |y| from 0.270 to 0.370 m (its alley), else 0. Score = the two sides' sum. Read from the state at hand (the last
    step)."""
    return _TRACKER.result([objects[n] for n in STICK_NAMES], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.028  # 56 mm: 18 mm either side of the 20 mm post (the foot below spans 40 mm; the pad tips stay above it)
GRIP = -(L.POST_SIZE[0] - 0.008) / 2  # 12 mm commanded across the 20 mm post
YAW = 0.0  # both arms: the fingers close along y on the post's two y faces (a square post: any heading would do)
GRIP_UP = 0.050  # the TCP 50 mm above the foot's underside: pad tips 37 mm up, the pads on the post's upper 34 mm
DROP = 0.001  # each stick is let go with its foot 1 mm above the court
PICK_Z = L.COURT_TOP + GRIP_UP
PLACE_Z = L.COURT_TOP + DROP + GRIP_UP
APPROACH = 0.030  # each final descent starts this far above its end height and is slow
HOVER_Z = L.TOP + 0.120  # travel height: a carried stick's foot 69 mm over the court (nothing on the way)
UP_Z = L.TOP + 0.140  # after letting go: the pad tips 55 mm over the stick's top
PARK = (0.200, 0.250, L.TOP + 0.160)  # each arm ends over its own side of the near court (y mirrored)
# Pre-compensation for the GPU run: in our runs carried parts landed about 4 mm short of the commanded point along the
# way they were carried, so each place point is commanded PLACE_LEAD past the ITF point along the carry. Offline (no
# lag) the stick is then set 2.0 mm beyond the net line and 3.4 mm outside the ITF point, inside the 6 and 8 mm
# windows; a 4 mm short landing puts it on the point. If off_net_mm or off_itf_mm in TASK_TRACE is off by more than
# 2 mm on the GPU, move that arm's PLACE_TRIM by minus the offset (x: off_net_mm; y: off_itf_mm, outward positive).
PLACE_LEAD = 0.004
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}  # (x, outward |y|) in m


def place_point(arm: str) -> tuple[float, float]:
    """Where the arm sets its stick down: the ITF point on its side, plus the lead along the carry, plus the trim."""
    sign = 1.0 if arm == "left" else -1.0
    name = L.NAMES[0] if arm == "left" else L.NAMES[1]
    sx, sy = L.START[name]
    tx, ty = L.NET_X, sign * L.STICK_Y
    d = math.hypot(tx - sx, ty - sy)
    ux, uy = (tx - sx) / d, (ty - sy) / d
    trim_x, trim_out = PLACE_TRIM[arm]
    return tx + PLACE_LEAD * ux + trim_x, ty + PLACE_LEAD * uy + sign * trim_out


def _plan(arm: str) -> list[Waypoint]:
    """Over its own side's stick, slowly down round the post, grip it, lift it, carry it over its ITF point, stand it
    slowly on the net line, let go, straight up, park."""
    sign = 1.0 if arm == "left" else -1.0
    name = L.NAMES[0] if arm == "left" else L.NAMES[1]
    x, y = L.START[name]
    px, py = place_point(arm)
    return [
        Waypoint(2.5, (x, y, HOVER_Z), OPEN, f"over {name}", YAW),
        Waypoint(1.0, (x, y, PICK_Z + APPROACH), OPEN, "down to just above the post", YAW),
        Waypoint(1.5, (x, y, PICK_Z), OPEN, "slowly down round the post", YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", YAW),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", YAW),
        Waypoint(2.0, (px, py, HOVER_Z), GRIP, "carry it over the net line", YAW),
        Waypoint(0.5, (px, py, HOVER_Z), GRIP, "settle over its place", YAW),
        Waypoint(1.0, (px, py, PLACE_Z + APPROACH), GRIP, "down to just above the court", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly stand it on the net line", YAW),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", YAW),
        Waypoint(1.0, (px, py, UP_Z), OPEN, "straight up, clear of it", YAW),
        Waypoint(1.5, (PARK[0], sign * PARK[1], PARK[2]), OPEN, "park", YAW),
    ]


def demo():
    """Both arms on one clock, each setting its own side's stick. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
