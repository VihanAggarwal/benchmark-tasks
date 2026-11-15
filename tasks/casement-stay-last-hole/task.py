"""On the station table, real size: a timber casement window (488 x 600 mm overall) standing across the table 42 cm in
front of the arm bases, hinged on the robot's left, opening outward (away from the robot). It rests open a little on
its peg stay (a Samuel Heath P4512-B 254 mm stay, from the maker's CAD): the stay's second hole is on the stay pin.
Two mechkit objects (layout.py): the window board with the two pins (fixed), and the window, ONE fixed-base
articulation: the frame (jambs and head), the sash on its hinge, the stay's swivel on the sash and the stay's bar on
its lift pin. Self-collision is off inside the articulation, so the board and the pins are a separate object: the bar
rests on the pins and the board. Nothing is latched or moved by code: the stay stays where gravity and the pins
leave it.

Goal: open the window as wide as its stay allows, the stay's last hole on the pin, and let go. Lift the stay off the
pin, swing the sash open while carrying the bar clear of the pin past the third hole, lower it onto the pin short of
the last hole and push on until the last hole drops over the pin. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=slide_from_start | release_on_pin_top | past_last | hold_at_last (see demo() below).
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built as articulations)
import rigstate  # noqa: E402
from offline_stay import OfflineStay  # noqa: E402
from scoring import BOARD, SASH, STAY, SWIVEL, WINDOW, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 20 steps

INSTRUCTION = "Open the window as wide as its stay allows, with the stay's last hole on the pin, and let go."

FRICTION = (0.35, 0.30)  # brass on brass, painted timber (est)

# -- masses (kg) ------------------------------------------------------------------------------------------------------
_H = L.SASH_Z1 - L.SASH_Z0
_TIMBER_V = 2 * L.STILE * L.SASH_T * _H + (L.SASH_W - 2 * L.STILE) * L.SASH_T * (L.STILE + L.BOTTOM_RAIL)
_PANE = (L.SASH_W - 2 * L.STILE + 0.02) * (_H - L.STILE - L.BOTTOM_RAIL + 0.02) * L.GLASS_T
SASH_MASS = L.TIMBER * _TIMBER_V + L.GLASS * _PANE  # about 3.1 kg: 1.7 kg of timber, 1.4 kg of 4 mm glass


# -- the objects ------------------------------------------------------------------------------------------------------
class _Offline(mk.Assembly):
    """A fixed part as the offline checker draws it: a small box of its own (``BOX``: centre and size, world), so the
    checker's overlap and carry proxies leave the mechanism alone. The simulator never uses this."""

    BOX = ((0.0, 0.0, 0.0), (0.01, 0.01, 0.01))

    def offline_box(self):
        return tuple(float(v) for v in self.BOX[0]), tuple(float(v) for v in self.BOX[1]), 0.0

    def offline_state(self, state):
        return self.rest_state()


class Window(_Offline):
    BOX = ((L.FRAME_X0 + L.FRAME_D / 2, 0.0, L.TOP + L.FRAME_H - L.MEMBER / 2),
           (L.FRAME_D, L.FRAME_W - 2 * L.MEMBER, L.MEMBER))  # the head


class Board(_Offline):
    BOX = ((L.BOARD_CX, 0.0, L.TOP + L.SILL_H / 2), (L.BOARD_X1 - L.BOARD_X0, L.FRAME_W, L.SILL_H))  # the slab


SASH_LINK = mk.hinge(SASH, "Z", L.SASH_ANCHOR, L.sash_shapes(), lower=L.SASH_LOWER, upper=L.SASH_UPPER,
                     mass=SASH_MASS, damping=L.SASH_DAMPING, friction=L.SASH_FRICTION)
SWIVEL_LINK = mk.hinge(SWIVEL, "Z", L.SWIVEL_ANCHOR, L.swivel_shapes(), lower=L.SWIVEL_LOWER, upper=L.SWIVEL_UPPER,
                       mass=L.SWIVEL_MASS, damping=L.SWIVEL_DAMPING, parent=SASH)
STAY_LINK = mk.hinge(STAY, "X", L.LIFT_ANCHOR, L.bar_shapes(), lower=L.LIFT_LOWER, upper=L.LIFT_UPPER,
                     mass=L.BAR_MASS, damping=L.LIFT_DAMPING, parent=SWIVEL)

WINDOW_SPEC = Window(name=WINDOW, pos=L.BASE_ORIGIN, color=L.WHITE, mass=None, yaw=L.BASE_YAW,
                     shapes=tuple(L.frame_shapes()), joints=(SASH_LINK, SWIVEL_LINK, STAY_LINK), friction=FRICTION,
                     **mk.TIGHT_FIT)
WINDOW_SPEC._check()
BOARD_SPEC = Board(name=BOARD, pos=(L.BOARD_CX, 0.0, L.TOP), color=L.BOARD, mass=None, yaw=0.0,
                   shapes=tuple(L.board_shapes()), friction=FRICTION, **mk.TIGHT_FIT)
BOARD_SPEC._check()
OBJECTS = [BOARD_SPEC, WINDOW_SPEC]


# -- scoring plumbing -------------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineStay()  # only stepped without a simulator (see offline_stay.py)


@mk.on_reset
def _reset_offline() -> None:
    OFFLINE.reset()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the window: mechkit has no environment AND the state handed in was not read
    from one (mechkit marks those ``source="sim"``)."""
    return mk.env() is None and getattr(objects.get(WINDOW), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    """Without a simulator nothing moves the window's joints: offline_stay.py moves them from the arms' commanded
    motion; the scorer then sees those states."""
    OFFLINE.step(arms)
    out = dict(objects)
    out[WINDOW] = mk.state_from_poses(WINDOW_SPEC, (*L.BASE_ORIGIN, *mk.yaw_quat(L.BASE_YAW)), np.zeros(6),
                                      joints=OFFLINE.joints(), source="offline")
    return out


def _r(v, n: int = 5) -> list:
    return [round(float(x), n) for x in np.asarray(v, dtype=float).ravel()]


def key_poses(objects) -> dict:
    """The dense pose trace of the key parts, every control step: the three joints, the ball, the stay's link pose,
    each hole's centre (web underside, world) and pin A's place in the bar's frame. Never raises."""
    try:
        sw = objects[WINDOW]
        r_s, p_s = sw.link_pose(STAY)
        q = (float(sw.joints[SASH]), float(sw.joints[SWIVEL]), float(sw.joints[STAY]))
        holes = sw.link_to_world(STAY, np.array([L.bar_point(s, L.WEB_BOT_H) for s in L.HOLE_S]))
        s_a, n_a, h_a = L.pin_in_bar(q, "A")
        return {"q_rad": _r(q), "ball": _r(sw.link_to_world(STAY, L.bar_point(L.TAIL_S, L.BALL_H))),
                "stay_link": _r(p_s), "stay_quat": _r(mk.matrix_to_quat(np.asarray(r_s, dtype=float))),
                "holes": [_r(h, 4) for h in holes],
                "pin_a_in_bar_mm": [round(s_a * 1000, 2), round(n_a * 1000, 2), round(h_a * 1000, 2)]}
    except Exception as exc:  # noqa: BLE001 - a debugging aid only
        return {"error": str(exc)[:200]}


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
    """The last hole on the pin, both grippers clear and everything still for the scorer's 20 steps; held 1 s. This
    runs every control step, so it also feeds the tracker (and, without a simulator, the offline kinematics)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    offline = no_simulator(objects)
    if offline:
        objects = offline_states(objects, arms)
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details, "pose": key_poses(objects)}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    if offline:
        row["offline"] = {"holder": OFFLINE.holder, "captured": OFFLINE.captured}
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 the last hole on the pin, let go and still; 0.5 the third hole on the pin, let go; 0.25 the third or last
    hole on the pin, still held; 0 otherwise (and after a hand enters the glass). The score of the latest step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. The right arm only; the
# left stays parked. Every pose of the stay below is the linkage's own (layout.q_for): pin A at a distance rho along
# the bar from the swivel, under its centre line, and the web's underside at a stated height over the pin.
# R1 grip the ball from above, pads on its two faces (across the bar), pad tips 4 mm below its centre.
# R2 lift: the web 6 mm over the pin's top (the second hole comes off the pin; the groove's legs 2.25 mm over it).
# R3 carry: rho from the second hole (105 mm) to 174 mm (12 mm short of the last hole), the web kept 6 mm over the
#    pin's top, so the third hole passes over the pin; the sash swings from 13.5 to 28.3 degrees; the hand turns with
#    the bar (23.6 degrees).
# R4 lower the bar onto the pin's top (the groove over it), commanded 2 mm lower (a light press).
# R5 push on to the last hole (rho 186 mm), still pressing, the hand led PUSH_LEAD ahead along the bar and PUSH_SIDE
#    across it, held 0.5 s: the hole drops over the pin; then back onto the centred pose and down onto the seat,
#    commanded 1 mm below it.
# R6 open, straight up 60 mm, away; hold.
# GPU tuning (job 20261115T014050): once gripped, the ball sat 10.3 to 10.4 mm below the TCP, not 9.0 (GRIP_SAG);
# while pushing along the pin's top the hand lagged 3.0 to 3.4 mm behind its command along the bar (the arm gives
# about 1 mm per newton against the pin-top friction and the hinge's), which left the pin 3.7 mm short of the last
# hole and, because the lag is along the bar and not along the ball's curved path, 1.05 mm off the centre line.
# PUSH_LEAD commands the hand that far ahead along the bar during the push only; nothing else changes.
# GPU tuning (job 20261115T023329, PUSH_LEAD 3 mm): the push ended with pin A at s 185.17 mm, n 1.06 mm in the bar
# (hole 4's centre 1.36 mm off the pin; the tip passes a hole only within 1.1 mm). n was pinned there by the groove's
# leg against the taper (1.04 mm of play): while pushing, the hand also lags about 1.0 mm to the right of the push
# (the hand's lag during the push was 2.6 to 3.4 mm in world -x, 0.2 to 0.7 mm in world -y, near-constant). From the
# trace's ball the residual is 0.85 mm along and 1.24 mm across: PUSH_LEAD 3.85 mm along and PUSH_SIDE 1.24 mm to
# the left of the push, then a 0.5 s dwell, still pressing with the lead held, so the hole can drop before the seat.
OPEN = -0.04695  # 94 mm
GRIP = -(2 * L.BALL_R - 0.008) / 2  # 6 mm commanded across the 14 mm ball
TCP_OVER_BALL = 0.013 - 0.004  # pad tips 13 mm below the TCP, 4 mm below the ball's centre
GRIP_SAG = 0.0013  # GPU: the gripped ball rides 1.3 mm lower in the hand than planned; the held moves allow for it
PUSH_LEAD = 0.00385  # GPU: the hand's lag along the bar while pushing on the pin's top (two runs: 3.65, then 0.85
# mm more with 3 mm of lead)
PUSH_SIDE = 0.00124  # GPU: and across it, to the left of the push (pin 1.06 mm across the bar with no side lead)
CLEAR = 0.006  # the web over the pin's top while carried (its legs then 2.25 mm over it)
PRESS = 0.002  # commanded below the pin's top while sliding
SEAT_PRESS = 0.001
RHO_LOWER = L.HOLE_S[3] - 0.012
HOVER = 0.060
PARK_R = (0.22, -0.26, L.TOP + 0.17)
VARIANTS = ("slide_from_start", "release_on_pin_top", "past_last", "hold_at_last")
RHO_PAST = L.HOLE_S[3] + 0.015  # past_last: carried 15 mm past the last hole


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q) -> float:
    """The gripper yaw that closes the pads across the bar (the pads on the ball's two faces)."""
    return _wrap(L.PHI_START + q[0] + q[1] - math.pi)


def tcp_of(q, up: float = 0.0, held: bool = True, lead: float = 0.0, side: float = 0.0):
    """The TCP for ball pose q: ``held`` allows for the ball's sag in the closed hand, ``lead`` puts the hand that far
    ahead of the ball along the bar, toward the swivel (the direction the hand pushes the bar in R5), ``side`` that
    far to the left of that direction in plan."""
    d = L.disc_world(q)
    x, y = float(d[0]), float(d[1])
    if lead or side:
        v = bar_back(q)
        x, y = (np.array([x, y]) + lead * v + side * np.array([-v[1], v[0]])).tolist()
    return (x, y, float(d[2] + TCP_OVER_BALL + up + (GRIP_SAG if held else 0.0)))


def bar_back(q) -> np.ndarray:
    """Unit vector in plan along the bar from the ball toward the swivel, world."""
    a, b = L.world_of_bar(q, [L.bar_point(L.HOLE_S[0], L.WEB_BOT_H), L.bar_point(L.HOLE_S[3], L.WEB_BOT_H)])
    v = (a - b)[:2]
    return v / float(np.linalg.norm(v))


def _w(duration, q, grip, label, up=0.0, held=True, lead=0.0, side=0.0) -> Waypoint:
    return Waypoint(float(duration), tcp_of(q, up, held, lead, side), grip, label, yaw_of(q))


def q_at(rho, web_over_pin_top=None, seat=False):
    if seat:
        return L.q_for(rho, L.PIN_SEAT_H - SEAT_PRESS)
    return L.q_for(rho, L.PIN_TOP_H + web_over_pin_top)


def grip_plan() -> list[Waypoint]:
    q0 = (0.0, 0.0, 0.0)
    return [
        _w(2.5, q0, OPEN, "over the stay's ball", up=HOVER, held=False),
        _w(1.5, q0, OPEN, "down beside the ball, pad tips 4 mm below its centre", held=False),
        _w(0.5, q0, OPEN, "settle", held=False),
        _w(1.5, q0, GRIP, "grip the ball on its two faces", held=False),
    ]


def lift_plan(rho0: float = L.HOLE_S[1]) -> list[Waypoint]:
    return [_w(0.8, q_at(rho0, CLEAR / 2 - 0.0060), GRIP, "start lifting the stay"),
            _w(1.2, q_at(rho0, CLEAR), GRIP, "lift: the second hole comes off the pin, the web 4 mm over its top"),
            _w(0.4, q_at(rho0, CLEAR), GRIP, "settle")]


def carry_plan(rho1: float, n: int = 8, seconds: float = 6.0, rho0: float = L.HOLE_S[1]) -> list[Waypoint]:
    out = []
    for i in range(1, n + 1):
        rho = rho0 + (rho1 - rho0) * i / n
        out.append(_w(seconds / n, q_at(rho, CLEAR), GRIP, "carry the bar clear over the pin, swinging the sash out"))
    out.append(_w(0.5, q_at(rho1, CLEAR), GRIP, "settle"))
    return out


def slide_plan(rho0: float, rho1: float, n: int = 4, seconds: float = 3.0) -> list[Waypoint]:
    out = [_w(1.5, q_at(rho0, -PRESS), GRIP, "lower the bar onto the pin's top, groove over it, a light press"),
           _w(0.5, q_at(rho0, -PRESS), GRIP, "settle")]
    pushed = rho1 != rho0
    lead, side = (PUSH_LEAD, PUSH_SIDE) if pushed else (0.0, 0.0)
    for i in range(1, n + 1):
        rho = rho0 + (rho1 - rho0) * i / n
        out.append(_w(seconds / n, q_at(rho, -PRESS), GRIP, "push on along the pin until the hole drops over it",
                      lead=lead, side=side))
    if pushed:
        out.append(_w(0.5, q_at(rho1, -PRESS), GRIP, "settle, still pressing: the hole drops", lead=lead, side=side))
    return out


def seat_plan(rho: float) -> list[Waypoint]:
    return [_w(1.5, q_at(rho, seat=True), GRIP, "down onto the seat, commanded 1 mm below it"),
            _w(0.5, q_at(rho, seat=True), GRIP, "settle")]


def release_plan(q) -> list[Waypoint]:
    return [_w(1.0, q, OPEN, "open: let go of the ball"),
            _w(1.0, q, OPEN, "straight up, clear of the stay", up=HOVER),
            Waypoint(1.5, PARK_R, OPEN, "away", yaw_of(q)),
            Waypoint(1.0, PARK_R, OPEN, "hold", yaw_of(q))]


def plans(variant: str = "") -> dict:
    """The right arm's plan for the demo or a counterfactual variant (the left stays parked)."""
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    plan = grip_plan() + lift_plan()
    if variant == "slide_from_start":  # pushed along the pin from the start: it drops into the third hole
        rho_end = L.HOLE_S[2] + 0.003
        plan += slide_plan(L.HOLE_S[1] + 0.013, rho_end, n=4, seconds=3.0)
        plan += seat_plan(L.HOLE_S[2])
        return {"right": plan + release_plan(q_at(L.HOLE_S[2], seat=True))}
    if variant == "past_last":  # carried past the last hole, lowered onto the pin's top there and let go
        plan += carry_plan(RHO_PAST, n=10, seconds=7.5)
        plan += slide_plan(RHO_PAST, RHO_PAST, n=1, seconds=0.5)
        return {"right": plan + release_plan(q_at(RHO_PAST, -PRESS))}
    plan += carry_plan(RHO_LOWER)
    if variant == "release_on_pin_top":  # lowered onto the pin's top and let go there, short of the last hole
        plan += slide_plan(RHO_LOWER, RHO_LOWER, n=1, seconds=0.5)
        return {"right": plan + release_plan(q_at(RHO_LOWER, -PRESS))}
    plan += slide_plan(RHO_LOWER, L.HOLE_S[3]) + seat_plan(L.HOLE_S[3])
    q_end = q_at(L.HOLE_S[3], seat=True)
    if variant == "hold_at_last":  # set right, never let go
        return {"right": plan + [_w(6.0, q_end, GRIP, "keep holding the ball")]}
    return {"right": plan + release_plan(q_end)}


def demo():
    """The right arm. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm checks).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      slide_from_start    lifted off the second hole and lowered straight back onto the pin, then pushed along it:
                          the third hole drops over the pin; let go there (0.5)
      release_on_pin_top  carried right, lowered onto the pin's top 12 mm short of the last hole and let go: the
                          bar rests on the pin, no hole on it, the sash free (0)
      past_last           carried 15 mm past the last hole, lowered onto the pin's top there and let go: the bar
                          rests on the pin, no hole on it, the sash free (0)
      hold_at_last        the demo without letting go: the last hole on the pin, still held (0.25)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
