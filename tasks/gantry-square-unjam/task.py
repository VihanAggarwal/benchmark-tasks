"""On the station table: two fixed steel guide rails (330 mm long, 12 mm wide, 40 mm tall) run along x from 17 to
50 cm in front of the arm bases, their inner faces 521 mm apart, each with a retaining lip along its top that
overhangs the gap by 10 mm. Between them, at the back, lies a loose aluminium gantry beam (40 mm deep, 520 mm long,
30 mm tall, 0.25 kg), out of square: its right end is 8 mm farther from the arms than its left end. Yellow marks on
the table outside the rails, at x 22 cm, show where the beam's ends must stop. Rule: square the beam first, then
bring it to the marks by both ends together (a gripper at each end while it moves), never letting one end lead
the other by more than 12 mm (the beam racks and, at about 16 mm, jams between the rails); then let go of it.

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
# Still for 1 s with both grippers' finger zones at least 30 mm from the beam.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "The grey gantry beam rides between two guide rails and is out of square: its right end is 8 mm farther from "
    "you than its left end. First pull the right end toward you until the beam is square, then pull the beam toward "
    "you by both ends together until both ends are level with the yellow marks beside the rails. Never let one end "
    "lead the other by more than 12 mm, or the beam racks and jams. Then let go of it."
)


def _rail(side: str, sign: float) -> list[Box]:
    """A fixed guide rail along x and the retaining lip along the top of its inner face."""
    rail = Box(f"rail_{side}", pos=(L.RAIL_X, sign * L.RAIL_Y, L.TOP + L.RAIL_H / 2),
               size=(L.RAIL_LEN, L.RAIL_W, L.RAIL_H), color=L.STEEL)
    lip = Box(f"lip_{side}", pos=(L.RAIL_X, sign * L.LIP_Y, L.TOP + L.LIP_Z0 + L.LIP_H / 2),
              size=(L.RAIL_LEN, L.LIP_W, L.LIP_H), color=L.STEEL)
    return [rail, lip]


MARKS = [
    Box(f"mark_{side}", pos=(L.TARGET_X, sign * L.MARK_Y, L.TOP + L.MARK_SIZE[2] / 2), size=L.MARK_SIZE,
        color=L.YELLOW)
    for side, sign in (("left", 1.0), ("right", -1.0))
]
BEAM = Box(
    "beam",
    pos=(L.BEAM_X0, 0.0, L.TOP + L.BEAM_SIZE[2] / 2),
    size=L.BEAM_SIZE,
    mass=L.BEAM_MASS,
    color=L.ALUMINIUM,
    yaw=L.BEAM_YAW0,
)
OBJECTS = [*_rail("left", 1.0), *_rail("right", -1.0), *MARKS, BEAM]

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


def success(objects) -> bool:
    """Both end centres within 5 mm of the marks' line, lead now <= 2 mm and never over 4 mm since it was first
    squared (before it had moved 25 mm), never racked, popped, off the rails or moved more than 20 mm without a
    gripper at each end, both grippers' finger zones 30 mm or more from the beam; held still 1 s. This runs first every control step, so it also feeds the tracker (with
    the arms rigstate read for this step)."""
    beam = objects["beam"]
    arms = rigstate.arms()
    result = _TRACKER.update(beam, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "beam_pos": beam.pos, "beam_quat": beam.quat_xyzw}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any step, one end led the other by more than 12 mm, the beam popped up (lowest point over 5 mm
    above the table, or tilted over 5 deg) or left the rails; if its centre came more than 20 mm toward the marks
    (net) without a gripper at each end; or if it was never squared (lead <= 2 mm). Else 1.0
    both ends within 5 mm of the marks, lead <= 2 mm, squared before it moved 25 mm and never over 4 mm after,
    both grippers 30 mm clear; 0.75 both ends within 10 mm, squared first, never over 7 mm after; 0.5 centre within
    15 mm of the marks' line; 0.25 moved 30 % of the way; 0 otherwise."""
    return _TRACKER.result(objects["beam"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.BEAM_SIZE[0] - 0.008) / 2  # across the beam's 40 mm depth: 32 mm commanded, 20 N per finger
# Both grips close the fingers along the beam's depth (x): yaw -pi/2 left and +pi/2 right, the signs that keep
# joint6 well inside its limits, turned by the beam's starting yaw so the pads meet its faces square.
YAW_L, YAW_R = -math.pi / 2, math.pi / 2
GRIP_Y = 0.200  # each arm grips its own end 200 mm from the beam's centre (60 mm in from the end)
GRIP_Z = L.TOP + 0.018  # TCP 18 mm up: pad tips 5 mm above the table, the flat pads on the beam's lower 5-27 mm
ENTRY_Z = L.TOP + 0.040  # the last 22 mm down are taken slowly
HOVER_Z = L.TOP + 0.100
UP_Z = L.TOP + 0.150  # where the hands go at the end: the open finger zone 50 mm clear of the beam's top
_TAN0 = math.tan(L.BEAM_YAW0)
XL0 = L.BEAM_X0 - GRIP_Y * _TAN0  # 0.4409: the beam's centreline at y +0.200 at the start
XR0 = L.BEAM_X0 + GRIP_Y * _TAN0  # 0.4471: ... and at y -0.200
# The arms trail a pull. Without payload compensation each joint is a spring (80/80/80/10/10/10 N m/rad); the
# compliance model (J K^-1 J^T at these grips) puts a gripper about 7 mm behind its command per newton along x.
# The beam's sliding friction is 2.45 N (0.25 kg, friction 1.0; 2.9 N to start it), so each arm pulling its half
# trails about 9 mm, and the right arm alone turning the beam about the left grip (1.3-1.5 N) about 10 mm.
# Squaring: the right arm is commanded SQUARE_TRIM past square at its grip, and both wrists turn back to square. If the
# arm trails as modelled, its end barely moves in this step (the arm only loads up) and the beam squares in the
# first second of the two-arm pull, when the loaded right end breaks free first; the wrists, held square, keep it
# so ('squared' latches at <= 2 mm on the way, with the centre 1-3 mm along in a planar friction model of the two
# compliant arms, far inside the 25 mm 'squared first' window, whatever the trim from 0 to 3 mm). If the arm did
# not trail at all, the right end would end 2.0 mm ahead: square, 2 mm inside the 4 mm limit for 1.0. On the GPU
# read lead_mm, squared_step and squared_travel_mm in TASK_TRACE.
# GPU calibration (L40S, 2026-11-05): with a 1.5 mm trim the right gripper trailed its command by 3.2 mm while
# turning the beam and the lead only came down from 8.9 mm to 2.9 mm (the 2 mm 'squared' line was never reached);
# the lead changes about 1.3 mm per mm of grip difference (end faces 520 mm apart, grips 400 mm apart). A 4.5 mm
# trim turns the beam about 3 mm further, to about 1 mm past square (|lead| about 1 mm), leaving room for the
# 1.5 mm the lead grew during the two-arm pull.
SQUARE_TRIM = 0.0045
# The pull: both grips go 9 mm past the marks' line and come back 9 mm. Trailing about 9 mm, the beam stops near
# the line and the return only unloads the arms (pushing it back would take about 20 mm); a stiff arm carries it
# past and pushes it back to the line. The ends land within 5 mm of the line for any trail from 0 to 14 mm. Read
# left_end_off_mm and right_end_off_mm in TASK_TRACE.
# GPU calibration: with a 9 mm overshoot the grippers trailed 3.7 mm (left) and 6.6 mm (right) and the beam stopped
# with its ends 6.0 and 2.0 mm past the line (centre 4 mm past), and the right arm's larger trail let the lead grow
# by 1.5 mm during the pull. So the overshoot is 5 mm. Second run (4.5 mm trim, 1.2 mm bias): squared at step 204
# (lead 0.7 mm), ends 2.3 mm past and 1.4 mm short of the line, but the right gripper still trailed the left by about
# 2 mm more than its bias and the lead grew steadily to 3.7 mm during the pull (0.75). So the right arm's pull is
# commanded further than the left's. Third run (3.5 mm bias): the lead stayed at 0.6-1.6 mm through the pull and the
# ends stopped 2.7 and 1.2 mm past the line, but easing both grippers back to the line itself pushed the right end
# back about 1 mm (its gripper, the most loaded, stood 1 mm behind the line) and the lead ended at 2.5 mm (0.75). So
# the bias is 4.5 mm (about 1.3 mm less lead), and the grippers ease off, let go and rise 2.5 mm past the line, where
# the trailing grippers stand at the end of the pull, so unloading them does not push the beam.
# Fourth run (4.5 mm bias, ease to 2.5 mm past the line over 1 s): the lead was 0.4 mm at the end of the pull, but
# the right gripper, commanded 7 mm back in 1 s, carried on and pushed its end back 0.8 mm and the beam yawed to a
# 2.9 mm lead (0.75). So the ease-off takes 3 s and stops each gripper 1 mm short of where it stands (3 mm past the
# line: a 0.1 N pull, far under the beam's friction), so neither gripper ever pushes the beam back.
# Fifth run (both eased over 3 s to 3 mm past the line): the right end stayed put but the left gripper crept 2.2 mm
# on toward the arms and the lead ended at 2.9 mm (0.75). Unloaded while still gripping, the left arm settles about
# 1 mm ahead of its command and the right about 0.7 mm behind it, so each eases to its own point: from the fourth and
# fifth runs, the left stays put eased to 1.8 mm past the line and the right to 2.8 mm.
# Sixth run (per-arm ease points): the same again, the left end moving about 2 mm further than the right while the
# grippers unloaded (the lead went 0.4 -> 2.2 mm; it went 0.4 -> 2.9 in the fourth and fifth runs), and the left
# end stopped 4.3 mm past the line. The unload's effect is steady, so the pull leaves the right end about 1.7 mm
# AHEAD (1.6 mm more bias: the lead changes about 1.3 mm per mm of grip difference) for the unload to bring square,
# and the overshoot is 1.5 mm less, so the ends stop about 2.5 mm past the line rather than 4.3.
OVERSHOOT = 0.0035
RIGHT_PULL_BIAS = 0.0061
PAST_X = L.TARGET_X - OVERSHOOT
EASE_X = {"left": L.TARGET_X - 0.0003, "right": L.TARGET_X - 0.0028}
EASE_S = 3.0


def _approach(x: float, y: float, yaw: float) -> list[Waypoint]:
    return [
        Waypoint(2.5, (x, y, HOVER_Z), OPEN, "over the beam's end", yaw),
        Waypoint(2.0, (x, y, ENTRY_Z), OPEN, "down to just above it", yaw),
        Waypoint(1.5, (x, y, GRIP_Z), OPEN, "slowly down round its faces", yaw),
        Waypoint(0.5, (x, y, GRIP_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, GRIP_Z), GRIP, "grip the beam's end", yaw),
        Waypoint(0.5, (x, y, GRIP_Z), GRIP, "settle the grip", yaw),
    ]


def _finish(y: float, yaw: float, bias: float = 0.0, side: str = "left") -> list[Waypoint]:
    past = PAST_X - bias
    ease = EASE_X[side]
    return [
        Waypoint(9.0, (past, y, GRIP_Z), GRIP, "pull toward the arms with the other end, in step", yaw),
        Waypoint(0.5, (past, y, GRIP_Z), GRIP, "settle", yaw),
        Waypoint(EASE_S, (ease, y, GRIP_Z), GRIP, "ease the pull off slowly, to just short of where it stands", yaw),
        Waypoint(0.5, (ease, y, GRIP_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (ease, y, GRIP_Z), OPEN, "let go", yaw),
        Waypoint(0.5, (ease, y, GRIP_Z), OPEN, "settle", yaw),
        Waypoint(2.0, (ease, y, UP_Z), OPEN, "straight up, clear of the beam", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: grip the leading left end, hold it while the right end is squared up (turning the wrist with the
    beam), then pull in step with the right arm to the marks, let go, go up."""
    y, yaw0 = GRIP_Y, YAW_L + L.BEAM_YAW0
    return _approach(XL0, y, yaw0) + [
        Waypoint(2.0, (XL0, y, GRIP_Z), GRIP, "hold while the right end comes square", YAW_L),
        Waypoint(0.5, (XL0, y, GRIP_Z), GRIP, "settle", YAW_L),
    ] + _finish(y, YAW_L, 0.0, "left")


def _right() -> list[Waypoint]:
    """Right arm: grip the lagging right end, pull it toward the arms until the beam is square (commanded 4.5 mm
    past square: the arm trails), then pull in step with the left arm to the marks, let go, go up."""
    y, yaw0 = -GRIP_Y, YAW_R + L.BEAM_YAW0
    return _approach(XR0, y, yaw0) + [
        Waypoint(2.0, (XL0 - SQUARE_TRIM, y, GRIP_Z), GRIP, "pull the lagging end square", YAW_R),
        Waypoint(0.5, (XL0 - SQUARE_TRIM, y, GRIP_Z), GRIP, "settle", YAW_R),
    ] + _finish(y, YAW_R, RIGHT_PULL_BIAS, "right")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
