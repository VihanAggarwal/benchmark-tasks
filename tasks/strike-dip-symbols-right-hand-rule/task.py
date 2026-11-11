"""On the station table: a geological field map (1:10,000) printed on a pale buff sheet, north at the top (+x, away
from the arms) with a north arrow, and three outcrop stations circled on it and labelled A (left, +y), C (middle) and
B (right, -y). Three loose T-shaped tokens lie on the map below the stations: strike-and-dip map symbols, each a long
dark red bar with a short yellow dip tick at its middle. The readings, strike/dip by the right-hand rule (the dip
direction is 90 degrees clockwise from the strike), are A 030/40, B 250/25 and C 135/60. Rule: plot them as a
geological map does, one symbol centred on each station, its bar along the recorded strike and its tick pointing
down-dip (strike + 90 degrees), lying flat on the map and let go. Seen from above a bearing b is the world yaw -b, so
the bars lie at yaw -30, 110 and -135 (45) and the ticks point to yaw -120, +20 and +135.

Each token is a loose mechkit compound (mechkit.py, next to this file): one rigid body of two boxes, so the scorer can
read its pose (the bar's centre and which way the tick points). The scorer also reads the arms (pad-tip points and
openings) through rigstate.py, which wraps the embodiment's step and reset from this file's import, only to check
that each token was let go; neither changes anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the tokens are built as compound bodies)
import rigstate  # noqa: E402
from scoring import NAMES, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0) needs every symbol let go and still for 1 s, and must then hold 2 s. Should the arms ever be
# unreadable, the last symbol would count from the end of its slow descent while still gripped; settling and letting
# go take 0.5 + 1.5 s more, so with 2 s the trial still cannot end while the fingers are closed on it.
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "Geological field map, 1:10,000, north at the top (its arrow points away from the arms). Stations are circled: "
    "A left, C middle, B right. Readings, strike/dip in degrees by the right-hand rule (dip direction 90 degrees "
    "clockwise from strike): A 030/40, B 250/25, C 135/60. Each red token is a strike-and-dip symbol: a long bar "
    "with a short yellow dip tick at its middle. Plot them: one symbol centred on each circle, turned as a "
    "geological map shows its reading, lying flat; then let go."
)

MAP = Box("field_map", pos=((L.MAP_X0 + L.MAP_X1) / 2, (L.MAP_Y0 + L.MAP_Y1) / 2, L.TOP + L.MAP_T / 2),
          size=(L.MAP_X1 - L.MAP_X0, L.MAP_Y1 - L.MAP_Y0, L.MAP_T), color=L.MAP_BUFF)
MARKS = [Box(name, pos=(cx, cy, L.INK_Z_W), size=(sx, sy, L.INK_T), color=L.INK_BLACK, yaw=yaw)
         for name, ((cx, cy), (sx, sy), yaw) in L.ink_marks().items()]


def _token(name: str):
    x, y, yaw_deg = L.TOKENS[name]
    return mk.assembly(
        name,
        origin=(x, y, L.MAP_TOP_W),
        yaw=math.radians(yaw_deg),
        mass=L.TOKEN_MASS,
        color=L.BAR_RED,
        shapes=[mk.box(L.BAR_CENTER, L.BAR), mk.box(L.TICK_CENTER, L.TICK, color=L.TICK_YELLOW)],
    )


TOKENS = [_token(n) for n in NAMES]
OBJECTS = [MAP, *MARKS, *TOKENS]

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


def success(objects) -> bool:
    """1.0 now: at each station a symbol centred within 8 mm, its bar within 6 degrees of the strike and its tick
    within 6 degrees of the right-hand-rule dip direction, lying flat on the map, let go and still for 1 s; held
    SUCCESS_HOLD_S = 2 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate
    read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    states = {n: objects[n] for n in NAMES}
    result = _TRACKER.update(states, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    for n in NAMES:
        t = objects[n]
        row[f"{n}_source"] = getattr(t, "source", None)
        row[f"{n}_quat"] = getattr(t, "quat_xyzw", None)
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
    """Per station (A, B, C), the qualifying token nearest its centre (lying flat on the map, let go, still for 1 s,
    its bar's centre within 8 mm of the station's centre): 1/3 if its bar is within 6 degrees of the recorded strike
    and its tick within 6 degrees of the right-hand-rule dip direction; 1/6 if the bar is on the strike but the tick
    points the other way; else 0. Score = the sum. Read from the state at hand (the last step)."""
    return _TRACKER.result({n: objects[n] for n in NAMES}, rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Each token is gripped straight down at the bar's centre with the fingers closing along the bar, on its two 20 mm
# end faces (60 mm apart): the pads are 20 mm wide, so they stay on the end faces and clear of the tick, and the
# fingers centre the bar between them. At yaw psi the fingers close along heading psi - 90 degrees, so a token whose
# tick points yaw theta is gripped at a yaw psi = theta (mod 180). Finger targets: the joint value is minus half the
# commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # fully open, 94 mm: 17 mm either side of the bar's 60 mm length
GRIP = -(L.BAR[1] - 0.008) / 2  # 52 mm commanded across the 60 mm bar
GRIP_UP = 0.017  # the TCP this far above the token's bottom: pad tips 4 mm above it (and the map)
DROP = 0.001  # each token is let go with its bottom 1 mm above the map
SLOW = 0.020  # the last 20 mm down onto a token, or onto the map, are taken slowly
HOVER = L.MAP_TOP_W + 0.070  # travel height: a carried token's bottom 53 mm up (nothing on the map stands up)
PARK_Z = L.TOP + 0.160
PARK = {"left": (0.200, 0.250, PARK_Z), "right": (0.200, -0.250, PARK_Z)}  # each arm ends on its own side
# The carried token turns with the gripper, so the gripper turns exactly as far as the tick must: from the start's
# tick yaw to the station's dip yaw (a quarter turn for tokens 1 and 2, an eighth for token 3), every gripper yaw
# within 90 degrees of the target's bearing from that arm's base, so joint 6 keeps 41 degrees or more to spare
# (yamkin): token 1 -30 -> +60, token 2 +110 -> +20, token 3 0 -> -45.
PICK_YAW_DEG = {"token_1": -30.0, "token_2": 110.0, "token_3": 0.0}
# Where each arm takes which token, in order. The left arm plots A (token 1), then C (token 3); the right arm plots B
# (token 2) and parks before the left arm goes to the middle.
PLAN = {"left": (("token_1", "A"), ("token_3", "C")), "right": (("token_2", "B"),)}
# Trim of the place yaw, for the GPU run: in our runs parts carried in a gripper ended turned counter-clockwise from
# the commanded yaw by about +0.6 to +1.7 degrees, so the place yaw is set 1.15 degrees clockwise (the residual is
# then within 0.55 degrees either way; offline, with no such turn, the token is laid 1.15 degrees off, inside the 6
# degree windows). If tick_err_deg in TASK_TRACE at rest is more than 1 degree, move YAW_TRIM by minus the signed error.
YAW_TRIM_DEG = -1.15
# Carried parts landed about 4 mm short of the commanded point in our runs (the arm trails under load), so each place
# point leads its station's centre by LEAD along the carry; the 8 mm centre window covers either case. If off_station_mm
# at rest is over 4 mm, move that arm's PLACE_TRIM (x, y) by minus the offset (origin_mm minus the station's centre).
LEAD = 0.004
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def turn_deg(token: str, station: str) -> float:
    """How far the tick must turn (degrees, -180..180): from the token's start yaw to the station's dip yaw."""
    return math.degrees(math.remainder(math.radians(L.DIP_YAW_DEG[station] - L.TOKENS[token][2]), 2 * math.pi))


def place_point(token: str, station: str, arm: str) -> tuple[float, float]:
    """The bar's centre as commanded at a station: its centre, LEAD further along the carry, plus the arm's trim."""
    x0, y0, _ = L.TOKENS[token]
    sx, sy = L.STATIONS[station]
    d = math.hypot(sx - x0, sy - y0)
    tx, ty = PLACE_TRIM[arm]
    return sx + LEAD * (sx - x0) / d + tx, sy + LEAD * (sy - y0) / d + ty


def _move(token: str, station: str, arm: str, first: bool) -> list[Waypoint]:
    """Over a token, slowly down round it, grip it by its bar, lift it, carry it over its station while turning it,
    set it down slowly on the map, let go, straight up."""
    x, y, _ = L.TOKENS[token]
    pick_yaw = math.radians(PICK_YAW_DEG[token])
    set_yaw = pick_yaw + math.radians(turn_deg(token, station) + YAW_TRIM_DEG)
    px, py = place_point(token, station, arm)
    pick_z = L.MAP_TOP_W + GRIP_UP
    set_z = L.MAP_TOP_W + DROP + GRIP_UP
    return [
        Waypoint(2.5 if first else 2.0, (x, y, HOVER), OPEN, f"over {token}", pick_yaw),
        Waypoint(1.0, (x, y, pick_z + SLOW), OPEN, "down to just above it", pick_yaw),
        Waypoint(1.2, (x, y, pick_z), OPEN, "slowly down round its bar", pick_yaw),
        Waypoint(0.5, (x, y, pick_z), OPEN, "settle", pick_yaw),
        Waypoint(1.5, (x, y, pick_z), GRIP, f"grip {token} by its bar's ends", pick_yaw),
        Waypoint(0.3, (x, y, pick_z), GRIP, "settle the grip", pick_yaw),
        Waypoint(1.2, (x, y, HOVER), GRIP, "lift it straight up", pick_yaw),
        Waypoint(2.5, (px, py, HOVER), GRIP, f"carry it over station {station}, turning it", set_yaw),
        Waypoint(0.5, (px, py, HOVER), GRIP, "settle over the station", set_yaw),
        Waypoint(1.0, (px, py, set_z + SLOW), GRIP, "down to just above the map", set_yaw),
        Waypoint(1.5, (px, py, set_z), GRIP, "slowly lay it on the map", set_yaw),
        Waypoint(0.5, (px, py, set_z), GRIP, "settle", set_yaw),
        Waypoint(1.5, (px, py, set_z), OPEN, "let go of it", set_yaw),
        Waypoint(0.3, (px, py, set_z), OPEN, "settle", set_yaw),
        Waypoint(1.0, (px, py, HOVER), OPEN, "straight up, clear of it", set_yaw),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for k, (token, station) in enumerate(PLAN[arm]):
        wps += _move(token, station, arm, k == 0)
    return wps + [Waypoint(1.5, PARK[arm], OPEN, "park, clear of the map's stations", wps[-1].yaw)]


def demo():
    """Both arms on one clock: the left arm plots A then C, the right arm B then parks. DEMO_HOLD_ONLY=1: no plan
    (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the
    one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
