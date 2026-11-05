"""On the station table: a yellow proving unit (a known voltage source) with two terminals, P+ and P-, side by side
across the midline 13 cm in front of the arm bases; behind it a terminal block carrying the five terminals of an
isolated three-phase supply in a row along the midline, 50 mm apart (L1, L2, L3, N, PE, colour-coded); and a two-pole
voltage tester's two probes, a red bar on a rest to the left and a black bar on a rest to the right, each lying along
y with its metal tip toward the midline. The procedure for proving a supply dead: prove the tester on the proving
unit, test every pair of the five conductors (10 pairs), prove the tester again; each test needs one probe on each of
two terminals at once, so each gripper holds one probe. The tips must never come near each other, a probe that is
not held must never touch a terminal, and a probe is put down only on a rest.

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
# Still for 3 s with neither gripper holding a probe: longer than the demo's last open (1.5 s) and retreat (1.0 s),
# so the trial cannot end while a gripper is still closed on a probe.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Prove that the five terminals on the dark grey block (L1, L2, L3, N and PE) are dead, using the two-pole "
    "tester's red and black probes. Each probe's metal tip is the end that points toward the middle of the table. "
    "Hold one probe in each gripper by its handle; a test is both tips resting on the tops of two different "
    "terminals at the same time, held still for at least 1 second. First test the tester on the yellow proving "
    "unit's two terminals, then test every pair of the five terminals (10 tests), then test the tester on the "
    "proving unit again. Keep the two metal tips at least 2 cm apart at all times, never let go of a probe anywhere "
    "but its rest, and finish with both probes back on their rests."
)


def _post(name: str) -> list[Cylinder]:
    """A fixed terminal post: a coloured insulating collar on the base and a brass contact cap on top."""
    x, y = L.POSTS[name]
    tag = name.lower().replace("+", "_pos").replace("-", "_neg")
    return [
        Cylinder(f"post_{tag}_collar", pos=(x, y, L.BASE_TOP + L.COLLAR_H / 2), radius=L.POST_RADIUS,
                 height=L.COLLAR_H, color=L.COLLAR[name]),
        Cylinder(f"post_{tag}_cap", pos=(x, y, L.COLLAR_TOP + L.CAP_H / 2), radius=L.POST_RADIUS, height=L.CAP_H,
                 color=L.BRASS),
    ]


PROVING_UNIT = Box("proving_unit", pos=(L.PROVE_X, 0.0, L.TOP + L.BASE_H / 2), size=L.PROVE_BASE_SIZE,
                   color=L.YELLOW)
TERMINAL_BLOCK = Box("terminal_block", pos=(L.BLOCK_X, 0.0, L.TOP + L.BASE_H / 2), size=L.BLOCK_SIZE,
                     color=L.BLOCK_GREY)
POSTS = [c for n in L.POSTS for c in _post(n)]
RESTS = [Box(f"{n.split('_')[0]}_rest", pos=(L.REST_X, L.REST_Y[n], L.TOP + L.REST_SIZE[2] / 2), size=L.REST_SIZE,
             color=L.REST_GREY) for n in L.PROBES]
RED_PROBE = Box("red_probe", pos=(L.REST_X, L.REST_Y["red_probe"], L.PROBE_START_Z), size=L.PROBE_SIZE,
                mass=L.PROBE_MASS, color=L.RED)
BLACK_PROBE = Box("black_probe", pos=(L.REST_X, L.REST_Y["black_probe"], L.PROBE_START_Z), size=L.PROBE_SIZE,
                  mass=L.PROBE_MASS, color=L.BLACK)
OBJECTS = [PROVING_UNIT, TERMINAL_BLOCK, *POSTS, *RESTS, RED_PROBE, BLACK_PROBE]

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
    """Proved, all 10 pairs tested, proved again, both probes back on their rests and let go, no rule ever broken;
    held still 3 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate read
    for this step)."""
    red, black = objects["red_probe"], objects["black_probe"]
    arms = rigstate.arms()
    result = _TRACKER.update(red, black, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "red_pos": red.pos, "black_pos": black.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if at any moment the two metal tips came within 15 mm of each other, a metal tip touched two terminals at
    once, a probe that was not held touched a terminal, a probe was neither held nor on a rest for over 0.25 s, or a
    probe fell off the table; 0 also if the tester was never proved. Else 1.0 proved, all 10 pairs tested after it,
    proved again after the last of them, both probes on their own rests and neither held; 0.9 the same but not put
    back; otherwise 0.25 for the first prove plus 0.05 for each supply pair tested after it (0.75 for all 10)."""
    return _TRACKER.result(objects["red_probe"], objects["black_probe"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.PROBE_W - 0.008) / 2  # 18 mm commanded across the probe's 26 mm
# Both grips close the fingers along x, across the bar (yaw -pi/2 left, +pi/2 right, the signs that reach the
# midline), so the pads sit on the bar's front and back faces and its two ends stick out sideways, along y.
YAW = {"left": -math.pi / 2, "right": math.pi / 2}
PROBE_OF = {"left": "red_probe", "right": "black_probe"}
READY = {"left": (0.323, 0.305, L.TOP + 0.26), "right": (0.323, -0.305, L.TOP + 0.26)}
GRIP_UP = 0.004  # the probe is gripped with the pad tips 4 mm above its bottom: 22 mm of its 26 mm side held
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
TCP_OVER_BOTTOM = GRIP_UP + PAD_TIPS  # 17 mm: the TCP above the held probe's underside
PICK_Z = L.REST_TOP + TCP_OVER_BOTTOM  # gripping a probe lying on its rest
RELEASE_Z = PICK_Z + 0.002  # letting go with the probe's bottom 2 mm above its rest
ENTRY = 0.020  # the last 20 mm down to a grip or a release are a separate, slow (1 s) move
HOVER_Z = L.TOP + 0.10  # over a rest, before going down to a probe
# A tip is lowered onto a post with its underside commanded 1 mm above the post's top: never below it, so the arm
# never pushes on the fixed post. The arm does not compensate a held weight (about 8 mm down for 0.3 kg at 0.3 m
# reach): about 1.6 mm for the 60 g probe, gripped at its centre of mass, so the tip settles onto the post, which then
# carries at most the probe's own weight (0.6 N) while the arm springs back toward the command above it. With no sag
# at all the tip is 1 mm up, inside the 4 mm 'on the post' band with 3 mm to spare for an upward error (a 1 degree
# pitch of the probe in the grip moves its tip 2.4 mm).
CONTACT_GAP = 0.001
CONTACT_Z = L.POST_TOP + CONTACT_GAP + TCP_OVER_BOTTOM
OVER_Z = L.POST_TOP + 0.015 + TCP_OVER_BOTTOM  # moving between posts: the tip's underside 15 mm above the post tops
LIFT_S, LOWER_S, TEST_S = 0.6, 1.0, 2.0  # 14 mm up; 14 mm down onto the post (1.4 cm/s); held 2 s for the 1 s test
ACROSS_SPEED = 0.08  # m/s between posts


def _tcp(side: str, pin_xy, z: float) -> tuple[float, float, float]:
    """The TCP that puts the probe's contact point over ``pin_xy``: the gripper holds the probe at its middle, so the
    TCP is 140 mm from the contact point along y, on the arm's own side."""
    s = 1.0 if side == "left" else -1.0
    return (float(pin_xy[0]), float(pin_xy[1]) + s * L.PIN_FROM_CENTRE, z)


def _across_s(a, b) -> float:
    """Time for a move between two posts: at 8 cm/s, rounded up to 0.5 s, at least 1 s."""
    d = math.dist(a, b)
    return max(1.0, math.ceil(d / ACROSS_SPEED * 2 - 1e-9) / 2)


class _Plan:
    """Both arms' waypoints on one clock."""

    def __init__(self) -> None:
        self.wps = {"left": [], "right": []}
        self.pos = dict(READY)
        self.fin = {"left": OPEN, "right": OPEN}
        self.pin = {"left": None, "right": None}  # the post each tip is on or over (name), if any

    def go(self, side, dur, pos, finger=None, label=""):
        finger = self.fin[side] if finger is None else finger
        self.wps[side].append(Waypoint(dur, tuple(pos), finger, label, YAW[side]))
        self.pos[side], self.fin[side] = tuple(pos), finger

    def wait(self, side, dur, label):
        dur = round(dur, 6)
        if dur > 0:
            self.go(side, dur, self.pos[side], None, label)

    def span(self, side) -> float:
        return sum(w.duration for w in self.wps[side])

    def over(self, side, post):
        """Lift the tip off the post it is on and carry it over ``post``."""
        here = L.POSTS[self.pin[side]]
        self.go(side, LIFT_S, _tcp(side, here, OVER_Z), label=f"lift the tip off {self.pin[side]}")
        self.go(side, _across_s(here, L.POSTS[post]), _tcp(side, L.POSTS[post], OVER_Z),
                label=f"carry it over {post}")
        self.pin[side] = post

    def lower(self, side):
        post = self.pin[side]
        self.go(side, LOWER_S, _tcp(side, L.POSTS[post], CONTACT_Z), label=f"lower the tip onto {post}")
        self.go(side, TEST_S, self.pos[side], label=f"hold the tip on {post}")

    def alone(self, side, step, label):
        """``step()`` adds waypoints to ``side``; the other arm keeps still for as long."""
        other = "right" if side == "left" else "left"
        t0 = self.span(side)
        step()
        self.wait(other, self.span(side) - t0, label)

    def move(self, side, post):
        """One probe moves to the next post while the other keeps its tip where it is."""
        other = "right" if side == "left" else "left"
        self.alone(side, lambda: (self.over(side, post), self.lower(side)), f"keep the tip on {self.pin[other]}")


def _build() -> _Plan:
    p = _Plan()
    for side in ("left", "right"):  # pick both probes up off their rests, at the same time (mirror images)
        name = PROBE_OF[side]
        x, y = L.REST_X, L.REST_Y[name]
        p.go(side, 2.0, (x, y, HOVER_Z), OPEN, f"over the {name.replace('_', ' ')}")
        p.go(side, 1.0, (x, y, PICK_Z + ENTRY), label="down to 2 cm above the grip")
        p.go(side, 1.0, (x, y, PICK_Z), label="slowly down beside it")
        p.go(side, 0.5, (x, y, PICK_Z), label="settle")
        p.go(side, 1.5, (x, y, PICK_Z), GRIP, "grip the probe at its middle")
        p.go(side, 1.0, (x, y, OVER_Z), label="lift it off the rest")
    for side, post in (("left", "P+"), ("right", "P-")):  # prove the tester on the proving unit
        p.go(side, 1.5, _tcp(side, L.POSTS[post], OVER_Z), label=f"carry the tip over {post}")
        p.pin[side] = post
        p.lower(side)
    # To the supply: each probe moves over its first post in turn, then both tips go down together (so the proving
    # unit is never paired with a supply terminal).
    p.alone("right", lambda: p.over("right", "L2"), "keep the tip on P+")
    p.alone("left", lambda: p.over("left", "L1"), "wait over L2")
    for side in ("left", "right"):
        p.lower(side)
    # The other nine pairs, one probe moving at a time. The red (left) tip is always on a post nearer the arms than
    # the black (right) tip, so neither ever passes the other: L1-L2, L1-L3, L2-L3, L2-N, L3-N, L3-PE, N-PE, L2-PE,
    # L1-PE, L1-N.
    for side, post in (("right", "L3"), ("left", "L2"), ("right", "N"), ("left", "L3"), ("right", "PE"),
                       ("left", "N"), ("left", "L2"), ("left", "L1"), ("right", "N")):
        p.move(side, post)
    # Prove again: the red tip goes over P+ first, then the black over P-, then both tips go down together.
    p.alone("left", lambda: p.over("left", "P+"), "keep the tip on N")
    p.alone("right", lambda: p.over("right", "P-"), "wait over P+")
    for side in ("left", "right"):
        p.lower(side)
    for side in ("left", "right"):  # both probes back on their rests, let go, up
        name = PROBE_OF[side]
        x, y = L.REST_X, L.REST_Y[name]
        p.go(side, LIFT_S, _tcp(side, L.POSTS[p.pin[side]], OVER_Z), label="lift the tip off the proving unit")
        p.go(side, 1.5, (x, y, OVER_Z), label="carry the probe over its rest")
        p.go(side, 0.6, (x, y, RELEASE_Z + ENTRY), label="down to 2 cm above the rest")
        p.go(side, 1.0, (x, y, RELEASE_Z), label="slowly down onto the rest")
        p.go(side, 0.5, (x, y, RELEASE_Z), label="settle")
        p.go(side, 1.5, (x, y, RELEASE_Z), OPEN, "let go")
        p.go(side, 1.0, (x, y, HOVER_Z), label="up")
    return p


def _left() -> list[Waypoint]:
    """Left arm: the red probe (P+ at each prove)."""
    return _build().wps["left"]


def _right() -> list[Waypoint]:
    """Right arm: the black probe (P- at each prove)."""
    return _build().wps["right"]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
