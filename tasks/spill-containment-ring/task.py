"""On the station table: a dark spill (a thin disc 120 mm across, fixed) on the midline 28 cm in front of the arm
bases, and two yellow spill-kit trays (fixed boards, 8 mm thick), one in front of each arm, each holding two white
absorbent socks (loose bars 200 x 30 x 30 mm, lying along x). The four socks must be laid flat on the table round
the spill, end to end, so that no gap in the ring is wider than 10 mm and every sock is within 60 mm of the spill's
edge, while nothing ever comes over the spill: no sock and no gripper's fingers may come within 5 mm of its edge,
seen from above, at any height.

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
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with all four socks placed in a closed ring and both grippers let go (each open, opening >= 0.9,
# or its finger zone more than 10 mm from every sock): a trial cannot end while a gripper still holds a sock.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Ring the dark spill with the four white absorbent socks from the yellow trays: lay them flat on the table end "
    "to end round it, close to its edge, leaving no gap wider than 1 cm. Never let a sock or a gripper's fingers "
    "come over the spill or within 5 mm of its edge, even up in the air."
)

SPILL = Cylinder("spill", pos=(L.SPILL_X, L.SPILL_Y, L.TOP + L.SPILL_H / 2), radius=L.SPILL_R, height=L.SPILL_H,
                 color=L.SPILL_DARK)


def _tray(name: str, spec) -> Box:
    (x, y), (sx, sy) = spec
    return Box(name, pos=(x, y, L.TOP + L.TRAY_T / 2), size=(sx, sy, L.TRAY_T), color=L.TRAY_YELLOW)


TRAYS = [_tray("left_tray", L.LEFT_TRAY), _tray("right_tray", L.RIGHT_TRAY)]
SOCK_OBJECTS = [
    Box(name, pos=(x, y, L.TRAY_TOP_Z + L.SOCK_H / 2), size=L.SOCK_SIZE, mass=L.SOCK_MASS, color=L.SOCK_WHITE)
    for name, (x, y) in L.SOCK_START.items()
]
OBJECTS = [SPILL, *TRAYS, *SOCK_OBJECTS]

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


def _socks(objects) -> dict:
    return {name: objects[name] for name in L.SOCKS}


def success(objects) -> bool:
    """All four socks flat on the table within 60 mm of the spill's edge in a ring round it with every joint 10 mm
    or less, both grippers let go, nothing ever over the spill; held still 1 s. This runs first every control step,
    so it also feeds the tracker (with the arms rigstate read for this step)."""
    socks = _socks(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(socks, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "sock_pos": {n: s.pos for n, s in socks.items()}}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, a sock or a gripper's finger zone came over the spill or within 5 mm of its edge (seen
    from above, at any height). Else 1.0 all four socks placed (flat on the table, within 60 mm of the spill's edge)
    and together in a ring round it with every joint 10 mm or less, both grippers let go; 0.75 all four placed and
    together round it with joints of 25 mm or less, at most one over 10 mm (or the 1.0 ring with a gripper still
    holding a sock); 0.5 three or more placed socks, linked into one group by gaps of 25 mm or less, wrapping at
    least 250 degrees round the spill; 0.25 at least two socks placed; 0 otherwise."""
    return _TRACKER.result(_socks(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap.
PART = -0.025  # 50 mm: 10 mm clear of each side of a sock; the finger zone keeps 12 mm outside the keep-out circle
GRIP = -(L.SOCK_W - 0.008) / 2  # 8 mm narrower than the sock's 30 mm width
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.006  # socks are gripped with the finger tips 6 mm above their bottom: 22 mm of the 30 mm side held
HOLD = GRIP_UP + PAD_TIPS  # the TCP is this far above a held sock's bottom
TRAY_GRIP_Z = L.TRAY_TOP_Z + HOLD
CARRY_Z = L.TOP + 0.060 + HOLD  # a carried sock's bottom 60 mm up: 22 mm over a sock on a tray, 30 mm over the ring
PLACE_Z = L.TOP + 0.002 + HOLD  # let go with the sock's bottom 2 mm above the table
ENTRY_Z = PLACE_Z + 0.020  # the last 20 mm are lowered slowly
UP_Z = L.TOP + 0.100  # after letting go: the finger zone 17 mm clear of the sock (more than 10 mm: let go)
# Yaws: 0 closes the fingers along y (across a sock lying along x); a quarter turn closes them along x. Near the
# midline the left arm needs -pi/2 and the right arm +pi/2 (the signs joint6 allows there).
TURN = {"left": -math.pi / 2, "right": math.pi / 2}
PLAN = {  # arm -> [(sock, its place in the ring)], in order
    "left": [("sock_1", "left"), ("sock_2", "back")],
    "right": [("sock_3", "right"), ("sock_4", "front")],
}


def _grip_point(sock: str, place: str) -> tuple[float, float]:
    """Where the demo grips a sock on its tray: the point that ends at its ring grip point (40 mm from the sock's
    centre toward its corner end). A sock laid along x is gripped 40 mm ahead of or behind its centre, as its place
    needs; one turned a quarter turn is gripped 40 mm behind its centre, so the turn swings that offset across."""
    sx, sy = L.SOCK_START[sock]
    gx, _ = L.RING_GRIP[place]
    return gx, sy


def _lay(arm: str, sock: str, place: str) -> list[Waypoint]:
    """Pick a sock off its tray, carry it (turning it first if its place lies along y) and lay it in the ring."""
    sx, sy = _grip_point(sock, place)
    gx, gy = L.RING_GRIP[place]
    yaw0 = 0.0
    yaw1 = TURN[arm] if abs(L.RING[place][2]) > 0.1 else 0.0
    wps = [
        Waypoint(1.8, (sx, sy, CARRY_Z), PART, f"over {sock}", yaw0),
        Waypoint(1.2, (sx, sy, TRAY_GRIP_Z), PART, "down beside it", yaw0),
        Waypoint(0.5, (sx, sy, TRAY_GRIP_Z), PART, "settle", yaw0),
        Waypoint(1.2, (sx, sy, TRAY_GRIP_Z), GRIP, f"grip {sock}", yaw0),
        Waypoint(1.0, (sx, sy, CARRY_Z), GRIP, "lift it off the tray", yaw0),
    ]
    if yaw1 != yaw0:
        wps.append(Waypoint(1.2, (sx, sy, CARRY_Z), GRIP, "turn it a quarter turn over the tray", yaw1))
    wps += [
        Waypoint(2.0 if yaw1 != yaw0 else 1.6, (gx, gy, CARRY_Z), GRIP, f"carry it to the {place} side of the spill",
                 yaw1),
        Waypoint(0.8, (gx, gy, ENTRY_Z), GRIP, "down to 2 cm above the table", yaw1),
        Waypoint(0.5, (gx, gy, ENTRY_Z), GRIP, "settle", yaw1),
        Waypoint(1.5, (gx, gy, PLACE_Z), GRIP, "lower the last 2 cm slowly", yaw1),
        Waypoint(0.5, (gx, gy, PLACE_Z), GRIP, "settle", yaw1),
        Waypoint(1.0, (gx, gy, PLACE_Z), PART, "let go (fingers to 50 mm)", yaw1),
        Waypoint(1.0, (gx, gy, UP_Z), PART, "straight up, clear of the sock", yaw1),
    ]
    return wps


def _arm(arm: str) -> list[Waypoint]:
    """Both of this arm's socks, in order. The arm ends straight above its last sock, its fingers at 50 mm: well
    clear of the spill and of the sock (let go); the trial ends there once the ring has held still for 1 s."""
    wps = []
    for sock, place in PLAN[arm]:
        wps += _lay(arm, sock, place)
    return wps


def demo():
    """Both arms on one clock, in parallel: each lays the two socks from its own tray (the side of the ring on its
    side of the spill first, then the end nearer its corner). DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
