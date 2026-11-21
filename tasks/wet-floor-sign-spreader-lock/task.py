"""On the station table, standing for a corridor floor: a common 24 in two-leaf A-frame wet-floor sign (yellow
polypropylene, 0.55 kg) stands folded, its ridge (the top hinge, 610 mm up) along x at x 350 mm, y 0, its leaves (280
wide) face to face, looking along the corridor (y). Each leaf has a moulded hand grip (a fin 20 mm across, 140 mm
proud) low on its outer face and a printed warning panel. The hinge opens to a moulded stop at 25 degrees between the
leaves (12.5 degrees each, feet 264 mm apart). On the +y side lies a flat blue spill mat (300 x 200 mm, near edge at
y 222 mm, long edge along x); beyond the work area stands the corridor's side wall.

Goal: draw the leaves apart, one hand on each leaf's grip, until the hinge is on its stop (the sign opens in place,
about its hinge line, its +y foot ending 40 to 140 mm from the patch's near edge), and let go: it stands by itself,
square to the patch, beside it, nothing over it, hands off.

The sign is one mechkit articulation (mechkit.py, next to this file): leaf A is the base, leaf B hangs on the hinge at
the top, stopped at 25 degrees. No software rule acts on it. The arms are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the sign is built as an articulation)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import SIGN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 2 s of let-go and stillness

INSTRUCTION = ("Open the folded wet-floor sign all the way and leave it standing by itself at the near edge of the "
               "spill, square to it and not on it.")


# -- the parts --------------------------------------------------------------------------------------------------------
def _leaf_shapes(side: int):
    """One leaf's colliders in the base frame as folded: panel, foot pad, hand grip, printed warning. ``side`` -1 leaf
    A (its outer face toward -y, the right arm), +1 leaf B (toward +y, the left arm). x runs along the ridge."""
    panel = mk.box((0.0, side * L.T / 2, (L.H + L.PAD_H) / 2), (L.W, L.T, L.H - L.PAD_H), L.YELLOW)
    pad = mk.box((0.0, side * L.PAD_D / 2, L.PAD_H / 2), (L.PAD_W, L.PAD_D, L.PAD_H), L.PAD_GREY)
    rib = mk.box((0.0, side * (L.T + L.RIB_P / 2), (L.RIB_Z0 + L.RIB_Z1) / 2), (L.RIB_W, L.RIB_P, L.RIB_Z1 - L.RIB_Z0),
                 L.RIB_YELLOW)
    warning = mk.box((0.0, side * (L.T + L.PANEL_T / 2), L.PANEL_Z), (L.PANEL_W, L.PANEL_T, L.PANEL_H), L.PANEL_BLACK)
    return [panel, pad, rib, warning]


LEAF_B = mk.hinge("leaf_b", "X", (0.0, 0.0, L.H), _leaf_shapes(+1), upper=L.Q_STOP, lower=0.0, mass=L.MASS_B,
                  damping=L.HINGE_DAMPING, friction=L.HINGE_FRICTION, color=L.YELLOW)
SIGN_PART = mk.assembly(SIGN, origin=(L.SX, L.SY, L.TOP), mass=L.MASS_A, shapes=_leaf_shapes(-1),
                        joints=[LEAF_B], color=L.YELLOW, friction=L.FRICTION, tight_fit=True)
PATCH = Box("patch", pos=(L.PATCH_XC, (L.PATCH_Y0 + L.PATCH_Y1) / 2, L.TOP + L.PATCH_T / 2),
            size=(L.PATCH_X1 - L.PATCH_X0, L.PATCH_Y1 - L.PATCH_Y0, L.PATCH_T), color=L.PATCH_BLUE)
WALL = Box("wall", pos=((L.WALL_X0 + L.WALL_X1) / 2, (L.WALL_Y0 + L.WALL_Y1) / 2, L.TOP + L.WALL_H / 2),
           size=(L.WALL_X1 - L.WALL_X0, L.WALL_Y1 - L.WALL_Y0, L.WALL_H), color=L.WALL_BEIGE)
OBJECTS = [WALL, PATCH, SIGN_PART]


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows nothing of the leaves' hinge. Offline only (never in the simulator), this model gives the
# scorer: a gripper closed on a leaf's grip (grip points: TCP within 12 mm across it along x, 15 mm along y, 20 mm in
# height) draws that leaf with it. With both grips held the leaves open symmetrically about the hinge line, which ends
# where the two hands' spacing says (stopped at the hinge's stop); with one held, the other foot stays put (it grips
# the table) and the whole sign shifts toward the pulling hand. A released sign stays as it is.
HOLD_OPENING = 0.4  # a gripper this far or less open counts as holding
HOLD_ACROSS, HOLD_OUT, HOLD_Z = 0.012, 0.015, 0.020


def _bisect(f, target, lo, hi, increasing=True):
    flo, fhi = f(lo), f(hi)
    if increasing:
        if target <= flo:
            return lo
        if target >= fhi:
            return hi
    else:
        if target >= flo:
            return lo
        if target <= fhi:
            return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sign_state(hy=L.SY, theta=0.0, hx=L.SX):
    """The sign opened symmetrically (``theta`` per leaf, both feet on the table, the hinge line at y ``hy``, its middle
    at x ``hx``), as the checker's and the tests' state."""
    th = float(theta)
    origin = (hx, hy - L.H * math.sin(th), L.TOP)
    pose = (*origin, *mk.quat_from_ypr(0.0, 0.0, -th))  # leaf A rolled about the ridge: its foot toward -y
    return mk.state_from_poses(SIGN_PART, pose, joints={"leaf_b": 2.0 * th}, source="offline")


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.hy = L.SY
        self.theta = 0.0
        self.hx = L.SX
        self.hold: dict = {}  # arm side -> "A" or "B"

    def _rib_world(self, leaf, theta=None, hy=None):
        n, z = L.rib_point(leaf, self.theta if theta is None else theta)
        return (self.hy if hy is None else hy) + n, L.TOP + z

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        tcps = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            tcp = np.asarray(tcp, dtype=float)
            held = None
            if opening <= HOLD_OPENING and abs(float(tcp[0]) - self.hx) <= HOLD_ACROSS:
                for leaf in ("A", "B"):
                    ry, rz = self._rib_world(leaf)
                    if abs(float(tcp[1]) - ry) <= HOLD_OUT and abs(float(tcp[2]) - rz) <= HOLD_Z:
                        held = leaf
            if held is None:
                self.hold.pop(side, None)
                continue
            self.hold[side] = held
            tcps[held] = tcp
        lim = L.THETA_LOCK
        if "A" in tcps and "B" in tcps:
            span = float(tcps["B"][1] - tcps["A"][1])
            self.theta = _bisect(lambda t: self._rib_world("B", t, 0.0)[0] - self._rib_world("A", t, 0.0)[0], span,
                                 0.0, lim)
            self.hy = (float(tcps["A"][1]) + float(tcps["B"][1])) / 2.0 - (
                self._rib_world("A", self.theta, 0.0)[0] + self._rib_world("B", self.theta, 0.0)[0]) / 2.0
        elif "A" in tcps:
            foot_b = self.hy + L.foot_n("B", self.theta)
            ty = float(tcps["A"][1])
            self.theta = _bisect(lambda t: foot_b - L.H * math.sin(t) + self._rib_world("A", t, 0.0)[0], ty, 0.0, lim,
                                 increasing=False)
            self.hy = foot_b - L.H * math.sin(self.theta)
        elif "B" in tcps:
            foot_a = self.hy + L.foot_n("A", self.theta)
            ty = float(tcps["B"][1])
            self.theta = _bisect(lambda t: foot_a + L.H * math.sin(t) + self._rib_world("B", t, 0.0)[0], ty, 0.0, lim)
            self.hy = foot_a + L.H * math.sin(self.theta)

    def states(self) -> dict:
        return {SIGN: sign_state(self.hy, self.theta, self.hx)}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(SIGN), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The sign opened to its stop (feet 244 to 270 mm apart, each leaf 9 to 16 degrees off vertical), its near foot 40
    to 140 mm from the patch, nothing over it, its middle within the patch's length, the ridge within 10 degrees of the
    patch's edge, both grippers 50 mm clear and the sign still for 2 s; held 1 s. This runs first every control step, so
    it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {SIGN: objects.get(SIGN)}
    arms = rigstate.arms()
    if no_simulator(parts):
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(SIGN), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the sign lies over (a leaf 45 degrees off vertical), is not still, or is not opened (spread under 70 mm);
    1.0 open to the stop, lean, placement, beside, square, still and hands off; 0.75 the same with a gripper within
    50 mm; 0.5 still, open to the stop, ridge within 30 degrees, nothing over the patch, otherwise; 0.25 still and
    opened otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
OPEN = -0.04695  # 94 mm
OPEN_G = -(L.RIB_W + 0.040) / 2  # 60 mm round the 20 mm grip: 20 mm each side
GRIP_G = -(L.RIB_W - 0.008) / 2  # 12 mm commanded across the grip
CLOSED = 0.0
YAW = {"A": math.pi / 2, "B": -math.pi / 2}  # both close along x, across the grip (each arm's nearer yaw)

HOVER = 0.155  # pad tips 22 mm over the grip's top
ABOVE = 0.105  # TCP height with the pad tips 13 mm lower: the open fingers straddle the grip from here down
OVERTRAVEL = math.radians(1.0)  # pull this much past the stop's angle (10 mm at the grip): the stop takes it up
ARC_STEPS = 4
OPEN_SECONDS = 8.0
PARK_YAW = 0.0  # the hands turn back to yaw 0 on the way aside (away from the wrist's limit)
PARK_L = (0.20, 0.40, 0.155)
PARK_R = (0.20, -0.40, 0.155)


def grip_xyz(leaf: str, theta: float):
    """The TCP at ``theta`` per leaf: the grip point (world) of the sign opened in place."""
    n, z = L.rib_point(leaf, theta)
    return L.SX, L.SY + n, L.TOP + z


def _w(d, p, finger, label, leaf, yaw=None):
    return Waypoint(d, tuple(float(v) for v in p), finger, label, YAW[leaf] if yaw is None else yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def arm_plan(leaf: str, theta_end: float, release: bool = True):
    """One arm's moves: over the leaf's grip, down round it, grip, draw it out along its arc to ``theta_end`` (per
    leaf), let go, up, aside."""
    p0 = grip_xyz(leaf, 0.0)
    over = (p0[0], p0[1], L.TOP + HOVER)
    near = (p0[0], p0[1], L.TOP + ABOVE)
    park = PARK_L if leaf == "B" else PARK_R
    plan = [
        _w(2.0, over, OPEN_G, f"over leaf {leaf}'s grip", leaf),
        _w(1.5, near, OPEN_G, "down to the grip's top", leaf),
        _w(1.5, p0, OPEN_G, "slowly down round the grip", leaf),
        _w(0.6, p0, OPEN_G, "settle", leaf),
        _w(1.5, p0, GRIP_G, "close on the grip", leaf),
        _w(0.5, p0, GRIP_G, "settle", leaf),
    ]
    for k in range(1, ARC_STEPS + 1):
        plan.append(_w(OPEN_SECONDS / ARC_STEPS, grip_xyz(leaf, theta_end * k / ARC_STEPS), GRIP_G,
                       f"draw the leaf out, arc {k} of {ARC_STEPS}", leaf))
    pe = grip_xyz(leaf, theta_end)
    plan.append(_w(1.0, pe, GRIP_G, "hold it against the hinge's stop", leaf))
    if release:
        plan += [
            _w(1.5, pe, OPEN_G, "let go", leaf),
            _w(0.5, pe, OPEN_G, "settle", leaf),
            _w(1.5, (pe[0], pe[1], L.TOP + HOVER), OPEN_G, "straight up", leaf),
            _w(1.5, (park[0], park[1], L.TOP + park[2]), OPEN, "aside", leaf, PARK_YAW),
        ]
    else:
        plan.append(_w(3.0, pe, GRIP_G, "keep holding", leaf))
    return plan


# Wrong ways (counterfactuals): where each leaf ends, and whether the hands let go.
VARIANT = {"short": (0.6 * L.THETA_LOCK, True), "held": (L.THETA_LOCK + OVERTRAVEL, False)}
VARIANTS = tuple(VARIANT)


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant, on one clock. The left arm takes leaf B (+y), the
    right arm leaf A (-y): each hand draws the leaf on its own side outward."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    theta_end, release = VARIANT.get(variant, (L.THETA_LOCK + OVERTRAVEL, True))
    return {"left": arm_plan("B", theta_end, release), "right": arm_plan("A", theta_end, release)}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short      each leaf drawn only 60 percent of the way: not on the stop, the sign stands part open (0.25)
      held       the right way, but both hands keep hold of the grips: a gripper within 50 mm (0.75)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
