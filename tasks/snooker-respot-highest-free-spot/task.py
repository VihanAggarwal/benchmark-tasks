"""On the station table: a 1:6 snooker table (a fixed green bed 296 x 595 mm with cushion rails round it), its long
axis along y with the baulk end at -y, white baulk line, D and six unlabelled spots. Seven loose pucks stand for the
balls on the table: reds on the black spot and on the pink spot, a third red, the pink and the blue in play away
from their spots, the green on its spot and the brown away from its spot. The black waits on a tray beyond the top
end (+y), the yellow on a tray beyond the baulk end (-y). Rule (WPBSA Section 3 Rule 7): re-spot each colour on its
own spot if free, otherwise on the highest-value spot available, not touching another ball, and without moving any
ball on the table. So the black goes on the blue spot and the yellow on its own spot, the right-hand corner of the
D seen from the baulk end. The scorer reads the arms (pad-tip points and openings) through rigstate.py, which wraps
the embodiment's step and reset from this file's import, only to check that each colour was let go.

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
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. A colour counts only once released (no pad-tip point lower than 10 mm above its top
# within the finger reach) and still for 1 s. Should the arms ever be unreadable, a colour would count 1 s after
# its slow set-down while still gripped; settling and opening take 0.5 + 1.5 s more, so with 2 s the trial cannot
# end while the fingers are closed on a colour.
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "You are the referee. After the foul, the black and the yellow have to be re-spotted; they are on the trays at "
    "the two ends of the table. Re-spot both as the rules of snooker require, without moving any other ball on the "
    "table, then let go. The discs stand in for the balls."
)

BED = Box("table_bed", pos=(L.CX, L.CY, L.TOP + L.BED_H / 2), size=(L.PLAY_W, L.PLAY_L, L.BED_H), color=L.CLOTH)
_RZ = L.TOP + L.RAIL_H / 2
RAILS = [
    Box("cushion_near", pos=(L.BED_X0 - L.RAIL_W / 2, L.CY, _RZ), size=(L.RAIL_W, L.PLAY_L + 2 * L.RAIL_W, L.RAIL_H),
        color=L.RAIL_GREEN),
    Box("cushion_far", pos=(L.BED_X1 + L.RAIL_W / 2, L.CY, _RZ), size=(L.RAIL_W, L.PLAY_L + 2 * L.RAIL_W, L.RAIL_H),
        color=L.RAIL_GREEN),
    Box("cushion_top", pos=(L.CX, L.BED_Y1 + L.RAIL_W / 2, _RZ), size=(L.PLAY_W, L.RAIL_W, L.RAIL_H),
        color=L.RAIL_GREEN),
    Box("cushion_bottom", pos=(L.CX, L.BED_Y0 - L.RAIL_W / 2, _RZ), size=(L.PLAY_W, L.RAIL_W, L.RAIL_H),
        color=L.RAIL_GREEN),
]
_MZ = L.MARK_TOP - L.MARK_T / 2  # every marking: 0.5 mm thick, its top 0.2 mm above the bed
MARKS = [Box("baulk_line", pos=(L.CX, L.BAULK_Y, _MZ), size=(L.PLAY_W, L.LINE_W, L.MARK_T), color=L.WHITE)]
MARKS += [Box(f"d_line_{k + 1}", pos=(x, y, _MZ), size=(length, L.LINE_W, L.MARK_T), yaw=yaw, color=L.WHITE)
          for k, (x, y, length, yaw) in enumerate(L.d_strips())]
MARKS += [Cylinder(f"spot_{name}", pos=(x, y, _MZ), radius=L.SPOT_R, height=L.MARK_T, color=L.WHITE)
          for name, (x, y) in L.SPOTS.items()]
TRAYS = [Box(f"tray_{name}", pos=(x, y, L.TOP + L.TRAY_H / 2), size=(L.TRAY, L.TRAY, L.TRAY_H), color=L.WOOD)
         for name, (x, y) in L.TRAYS.items()]


def _puck(name: str) -> Cylinder:
    x, y = L.START[name]
    return Cylinder(name, pos=(x, y, L.start_bottom(name) + L.PUCK_H / 2), radius=L.PUCK_R, height=L.PUCK_H,
                    mass=L.PUCK_MASS, color=L.COLOUR[name])


PUCKS = [_puck(n) for n in L.NAMES]
OBJECTS = [BED, *RAILS, *MARKS, *TRAYS, *PUCKS]
PUCK_NAMES = tuple(p.name for p in PUCKS)

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the pucks teleporting back)."""
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


def _pos(s) -> list | None:
    try:
        return [round(float(v) * 1000, 1) for v in s.pos]
    except (AttributeError, TypeError, ValueError):
        return None


def success(objects) -> bool:
    """1.0 now: no table puck ever more than 8 mm from its start, the black counting on the blue spot and the
    yellow on the yellow spot (centre within 8 mm, flat on the bed, touching no other puck, released, still for
    1 s); held SUCCESS_HOLD_S = 2 s. This runs first every control step, so it also feeds the tracker (with the
    arms rigstate read for this step)."""
    arms = rigstate.arms()
    pucks = [objects[n] for n in PUCK_NAMES]
    result = _TRACKER.update(pucks, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if isinstance(arms, dict):  # each arm on its own: a missing one is traced as None, never a crash
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    paths = _TRACKER.window_paths() if result.details.get("valid_state") else None
    row["pos_mm"] = {n: _pos(objects[n]) for n in PUCK_NAMES}
    row["path_mm"] = None if paths is None else {
        n: [round(p * 1000, 2) for p in paths[i]] for i, n in enumerate(PUCK_NAMES) if n in L.RESPOT}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any table puck has been more than 8 mm from its start at any step. Else black part + yellow part: the
    black 0.5 counting on the blue spot, 0.25 on any other spot; the yellow 0.5 counting on the yellow spot. A
    colour counts on a spot with its centre within 8 mm of it, flat on the bed, no other puck's centre within
    31 mm, released and still for 1 s. Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in PUCK_NAMES], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# The demo works out the black's spot itself, from the observed layout: walk the spots from the highest value down
# and take the first one with no puck centre within 31 mm (Section 2 Rule 18); the yellow's own spot is checked
# the same way. Offline and on the GPU the layout is fixed (object_jitter 0), so this reads layout.START.
def free_spot(colour: str, centres=None) -> str:
    """The colour's own spot if free, else the highest-value free spot (centres: name -> (x, y) of the table
    pucks; the layout's start by default)."""
    centres = {n: L.START[n] for n in L.TABLE} if centres is None else centres

    def free(spot: str) -> bool:
        sx, sy = L.SPOTS[spot]
        return all(math.hypot(x - sx, y - sy) > L.TOUCH for x, y in centres.values())

    if free(colour):
        return colour
    for spot in sorted(L.SPOTS, key=lambda k: -L.VALUE[k]):
        if free(spot):
            return spot
    raise ValueError("no free spot")


# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.028  # 56 mm: 13 mm either side of a puck (no other puck centre within 73 mm of a pick or a place point)
GRIP = -(2 * L.PUCK_R - 0.008) / 2  # 22 mm commanded across the 30 mm puck
YAW = 0.0  # both arms: the fingers close along y (a round puck can be gripped at any heading)
GRIP_UP = 0.017  # the TCP this far above the puck's bottom: the pad tips 4 mm above the bed or tray
DROP = 0.001  # each colour is let go with its bottom 1 mm above the bed
SLOW = 0.020  # the last 20 mm down onto a puck, or onto the bed, are taken slowly
HOVER_Z = L.TOP + 0.085  # travel height: a carried puck's bottom 68 mm up, 48 mm over the cushions
RETREAT_Z = L.TOP + 0.085  # straight up after the release: the pad tips 42 mm over the puck's top
PARK = (0.200, 0.250, L.TOP + 0.150)  # each arm ends over the near cushion at its own end (y mirrored)
LEAD = 0.004  # place points lead the target this far along the carry: parts land about 4 mm short
# Per-arm trim of the place points (x, y), for the GPU run: if a colour lands off its spot by a steady amount, read
# its pos_mm in TASK_TRACE and move that arm's place point by minus the offset.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
ARM_OF = {"black": "left", "yellow": "right"}  # each tray is reachable only by the arm on its side
# The arms take turns over the table: the right arm sets the yellow down first; the left arm, the black already in
# its fingers, waits over its tray and carries only once the right arm has let go and risen (their elbows would
# pass within 1 cm if both were over their spots at once).
CARRY_WAIT = {"left": 8.0, "right": 0.0}


def place_point(colour: str) -> tuple[float, float]:
    """The place command for a colour: its spot, led 4 mm along the carry from its tray, plus the arm's trim."""
    sx, sy = L.SPOTS[free_spot(colour)]
    x0, y0 = L.START[colour]
    n = math.hypot(sx - x0, sy - y0)
    tx, ty = PLACE_TRIM[ARM_OF[colour]]
    return sx + LEAD * (sx - x0) / n + tx, sy + LEAD * (sy - y0) / n + ty


def _plan(colour: str) -> list[Waypoint]:
    """Over the colour on its tray, slowly down round it, grip it, lift it, carry it over its spot, settle, set it
    down slowly, let go, straight up, park. The left arm waits CARRY_WAIT s over its tray before carrying."""
    arm = ARM_OF[colour]
    x, y = L.START[colour]
    pick_z = L.TRAY_TOP + GRIP_UP
    place_z = L.BED_TOP + DROP + GRIP_UP
    px, py = place_point(colour)
    sign = 1.0 if arm == "left" else -1.0
    wait = [Waypoint(CARRY_WAIT[arm], (x, y, HOVER_Z), GRIP, "wait over the tray for the other arm", YAW)]
    return [
        Waypoint(2.0, (x, y, HOVER_Z), OPEN, f"over the {colour}", YAW),
        Waypoint(1.0, (x, y, pick_z + SLOW), OPEN, "down to just above it", YAW),
        Waypoint(1.5, (x, y, pick_z), OPEN, "slowly down round it", YAW),
        Waypoint(0.5, (x, y, pick_z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, pick_z), GRIP, f"grip the {colour}", YAW),
        Waypoint(0.3, (x, y, pick_z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", YAW),
        *(wait if CARRY_WAIT[arm] > 0 else []),
        Waypoint(2.5, (px, py, HOVER_Z), GRIP, "carry it over its spot", YAW),
        Waypoint(0.5, (px, py, HOVER_Z), GRIP, "settle over the spot", YAW),
        Waypoint(1.2, (px, py, place_z + SLOW), GRIP, "down to just above the bed", YAW),
        Waypoint(1.5, (px, py, place_z), GRIP, "slowly set it on the spot", YAW),
        Waypoint(0.5, (px, py, place_z), GRIP, "settle", YAW),
        Waypoint(1.5, (px, py, place_z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (px, py, place_z), OPEN, "settle", YAW),
        Waypoint(1.0, (px, py, RETREAT_Z), OPEN, "straight up, clear of it", YAW),
        Waypoint(1.5, (PARK[0], sign * PARK[1], PARK[2]), OPEN, "park at its own end", YAW),
    ]


def demo():
    """Both arms on one clock: the left arm re-spots the black, the right arm the yellow. DEMO_HOLD_ONLY=1: no plan
    (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the
    one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {ARM_OF[c]: _plan(c) for c in L.RESPOT}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
