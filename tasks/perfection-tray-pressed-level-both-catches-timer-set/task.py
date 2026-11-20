"""On the station table, real size: a Hasbro Gaming Perfection game unit (25 shapes, 60 s timer) stands centred at
x 330 mm, y 0, its front border (the timer dial and the STOP/START switch) toward the robot. The last game ended with
the pop: the shape tray stands 25 mm proud on its springs, the dial has run out to 0 and the switch is still at START,
which keeps the catches knocked open. Reset it for the next player, the way the instructions go: switch to STOP, turn
the dial clockwise to its stop (60 s), press the tray down level until both side catches hold it, slide the switch to
START so the clock runs.

Goal: the tray held down by both catches, the dial at 50 s or more and running (switch at START), both hands off.

The unit is one fixed mechkit articulation (mechkit.py, next to this file): the case, the tray on a slide down
(springs, preloaded) with a small rock about x, the dial on a hinge, the switch on a slide. The two catches (a sprung
pawl inside each side wall) are their own fixed part (links of one assembly never collide with each other, and each
pawl's tooth must meet the tray's barb). The tray is held down only by the barbs against the teeth. Declared rules
(mk.on_step): the springs' preload (a drive target above the top stop), the clockwork (the dial holds where it is
left at STOP and runs down 5 deg/s at START), the switch's two detents, and the run-out cam (at START with the dial
at 0 the pawls are driven open). The arms are read through rigstate.py (the let-go check).

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
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import CATCHES, DIAL, PAWL, ROCK, SWITCH, TRAY, UNIT  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold 1 s on top of the scorer's own 1 s of let-go and running clock

INSTRUCTION = "Reset the Perfection game for the next player: tray down and latched, timer set to 60 seconds and started."

# -- the parts --------------------------------------------------------------------------------------------------------
SLIDE = mk.slider(TRAY, "-Z", P.slide_shapes(), upper=L.TRAVEL_MAX, lower=0.0, mass=0.02, spring=L.SPRING_K,
                  damping=L.SPRING_DAMP, friction=0.0, color=L.GREY)
ROCKER = mk.hinge(ROCK, "X", L.ROCK_ANCHOR, P.tray_shapes(), upper=L.ROCK_MAX, lower=-L.ROCK_MAX,
                  mass=L.TRAY_MASS, spring=L.ROCK_K, damping=L.ROCK_DAMP, friction=0.0, parent=TRAY,
                  color=L.TRAY_YELLOW)
DIAL_LINK = mk.hinge(DIAL, "-Z", (L.DIAL_X, L.DIAL_Y, L.DECK_Z), P.dial_shapes(), upper=L.DIAL_FULL, lower=0.0,
                     mass=L.DIAL_MASS, spring=L.DIAL_SPRING, damping=L.DIAL_DAMP, friction=L.DIAL_FRICTION,
                     color=L.DIAL_RED)
SWITCH_LINK = mk.slider(SWITCH, "Y", P.switch_shapes(), upper=L.SWITCH_STOP, lower=0.0, mass=L.SWITCH_MASS,
                        spring=L.SWITCH_K, damping=L.SWITCH_DAMP, friction=0.0, color=L.WHITE)
GAME = mk.assembly(UNIT, origin=L.ORIGIN, shapes=P.case_shapes(), joints=[SLIDE, ROCKER, DIAL_LINK, SWITCH_LINK],
                   color=L.CASE_BLUE, friction=(0.6, 0.5), solver_iterations=32)
PAWLS = [mk.hinge(PAWL[s], "X" if s > 0 else "-X", (L.TRAY_XC, s * L.PAWL_HINGE_Y, L.PAWL_HINGE_Z), P.pawl_shapes(s),
                  upper=L.PAWL_UPPER, lower=0.0, mass=L.PAWL_MASS, spring=L.PAWL_K, damping=L.PAWL_DAMP,
                  friction=0.0, color=L.GREY) for s in S.SIDES]
CATCH_PART = mk.assembly(CATCHES, origin=L.ORIGIN, shapes=P.catch_mount_shapes(), joints=PAWLS, color=L.GREY,
                         friction=(0.4, 0.3), tight_fit=True)
LID = mk.assembly("lid", origin=L.LID_ORIGIN, shapes=P.lid_shapes(), color=L.CASE_BLUE)  # fixed, not touched
OBJECTS = [GAME, CATCH_PART, LID]


# -- the declared rules (simulator only; mechkit writes the drive targets at the start of every step) ----------------
def kicked(switch: float, dial: float) -> bool:
    """The run-out cam: with the switch at START and the dial run out, it holds both pawls open."""
    return switch <= L.SWITCH_START_TOL and dial <= math.radians(0.5)


def dial_target(switch: float, dial: float) -> float:
    """The clockwork as one step of the offline stand-in (whose dial is its own state): at START the dial runs down
    5 deg/s; at STOP it stays where it was left."""
    if switch <= L.SWITCH_START_TOL and dial > 0.0:
        return max(0.0, dial - L.DIAL_RATE * L.DT)
    return dial


# The simulator's clockwork. A target recomputed from the measured dial (dial minus one step's run-down) leaves a spring
# error of about 0.0004 rad, 0.0001 N m, which never beats the dial's 0.015 N m friction: the dial stood still at START
# (GPU run 20261120T082751). So the clock keeps its own target: seeded from the dial when the switch reaches START,
# lowered 5 deg/s every step, never more than CLOCK_LAG behind the dial (a hand winding it at START drags it along),
# down to CLOCK_FLOOR (just under the 0 stop, so the spring still drives a nearly run-out dial onto it and the run-out
# cam can kick); cleared at STOP and at every reset.
CLOCK_LAG = 0.20  # rad: 0.04 N m of spring at most, well over the 0.015 N m friction
CLOCK_FLOOR = -0.10  # rad: 0.02 N m pressing a run-out dial onto its 0 stop
_CLOCK = {"target": None}


def clock_reset() -> None:
    _CLOCK["target"] = None


def clock_target(switch: float, dial: float) -> float:
    """The dial's drive target this step (simulator): at STOP where it is (it holds where left); at START the clock's
    own target, independent of the measured angle, running down DIAL_RATE."""
    if switch > L.SWITCH_START_TOL:
        _CLOCK["target"] = None
        return dial
    t = dial if _CLOCK["target"] is None else _CLOCK["target"]
    t = max(t - L.DIAL_RATE * L.DT, dial - CLOCK_LAG, CLOCK_FLOOR)
    _CLOCK["target"] = t
    return t


@mk.on_step
def _rules(states) -> None:
    g = states.get(UNIT)
    if g is None:
        return
    sw, dial = float(g.joints[SWITCH]), float(g.joints[DIAL])
    mk.set_targets(GAME, TRAY, -L.SPRING_PRE)  # preload: the springs push toward 12 mm above the top stop
    mk.set_targets(GAME, DIAL, clock_target(sw, dial))
    mk.set_targets(GAME, SWITCH, mk.detent_target(sw, (0.0, L.SWITCH_STOP), L.SWITCH_CAPTURE))
    target = L.PAWL_KICK if kicked(sw, dial) else 0.0
    for s in S.SIDES:
        mk.set_targets(CATCH_PART, PAWL[s], target)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy only carries loose parts; the unit is fixed and has no springs. Offline only (never in the
# simulator), this model gives the scorer what the physics would: closed pads below the tray's top on a side push that
# side down, each arm with at most F_MAX and a give of GIVE per newton; one side pressed alone rocks the tray to its
# 4 deg stop; a side's catch holds once its barb has passed under the tooth, unless the cam holds the pawls open;
# closed pads pinching the dial's grip bar turn the dial with the hand; closed pads beside the grip bar push it round;
# closed pads beside the switch knob slide it.
F_MAX = 8.0  # N, about what one arm pushes straight down (research brief)
GIVE = 0.0018  # m/N, the pad tips' give pressing down (wrist and arm)
CLOSED_GAP = 0.012
PAD_HALF = 0.006  # the closed pair is about 12 mm thick across its closing axis (along y at yaw 0)
PAD_HALF_W = 0.0105  # and 21 mm wide along x
PINCH_R = 0.004  # the demo pinches the grip bar this far out from the dial's axis (on the bar, along its length)
PINCH_BAND = (0.0015, 0.006)  # closed pads this far from the axis, at the bar's height, are read as a pinch
PINCH_SNAP = 0.35  # rad: and only when the line to them lies within this of the bar (else they are beside it)
Y_BARB = (L.BARB_Y_IN + L.BARB_Y_OUT) / 2
WEIGHT = L.TRAY_MASS * L.G
RAMP = (L.BARB_Z0 - L.BEVEL_HALF + (L.TOOTH_Y_IN - L.BEVEL_Y) - L.TOOTH_Z1, L.TRAVEL_SNAP)  # travel over which a barb rides its pawl out
PAWL_EXTRA = 0.7  # N per side on the ramp


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.q = {1: 0.0, -1: 0.0}
        self.caught = {1: False, -1: False}
        self.dial = 0.0
        self.switch = 0.0

    # -- the tray -----------------------------------------------------------------------------------------------
    def _press_depths(self, read) -> dict:
        """Per side, the deepest commanded travel of closed pads over that side of the tray (m below the popped top)."""
        out = {}
        for tcp, tips, opening in read:
            gap = opening * rigstate.FULL_GAP
            if gap > CLOSED_GAP:
                continue
            x, y = float(tips[0]) - L.UX - L.TRAY_XC, float(tips[1]) - L.UY
            if abs(x) > L.TRAY / 2 - PAD_HALF_W or abs(y) > L.TRAY / 2 - PAD_HALF or abs(y) < 0.030:
                continue
            d = (L.TOP + L.TRAY_TOP0) - float(tips[2])
            if d <= 0.0:
                continue
            s = 1 if y > 0 else -1
            out[s] = max(out.get(s, 0.0), d)
        return out

    def _tray(self, depths: dict) -> None:
        rock_span = 2 * Y_BARB * math.sin(L.ROCK_MAX)
        floor = {s: (L.TRAVEL_HOLD if self.caught[s] else 0.0) for s in S.SIDES}

        def extra(qc):
            return PAWL_EXTRA * sum(1 for s in S.SIDES if RAMP[0] <= self.q[s] < RAMP[1] and not self.caught[s])

        if len(depths) == 2:
            q = dict(self.q)
            for _ in range(30):  # each arm carries half the springs and gives under it, up to F_MAX
                qc = (q[1] + q[-1]) / 2
                f = (L.spring_force(qc) - WEIGHT + extra(qc)) / 2
                for s in S.SIDES:
                    qs = depths[s] - GIVE * max(0.0, min(f, F_MAX))
                    if f > F_MAX:  # the arm stalls: no deeper than where the springs take all it has
                        qs = min(qs, (2 * F_MAX + WEIGHT) / L.SPRING_K - L.SPRING_PRE)
                    q[s] = max(floor[s], qs)
            if abs(q[1] - q[-1]) > rock_span:
                lo = min(S.SIDES, key=lambda s: q[s])
                q[lo] = q[-lo] - rock_span
        elif len(depths) == 1:
            (s, d), = depths.items()
            f_need = L.spring_force(max(0.0, d - Y_BARB * math.sin(L.ROCK_MAX))) - WEIGHT + extra(d)
            qc_cap = (F_MAX + WEIGHT) / L.SPRING_K - L.SPRING_PRE
            qs = d - GIVE * max(0.0, min(f_need, F_MAX))
            qs = min(qs, qc_cap + Y_BARB * math.sin(L.ROCK_MAX)) if f_need > F_MAX else qs
            q = {s: max(floor[s], qs), -s: max(floor[-s], qs - rock_span)}
            if q[-s] > floor[-s] and self.caught[-s]:  # the other side held by its catch: the tray rocks about it
                q[-s] = floor[-s]
        else:
            q = {s: floor[s] for s in S.SIDES}
            for s in S.SIDES:  # one side caught, the other free: the free side rises to the rock stop
                if self.caught[-s] and not self.caught[s]:
                    q[s] = max(0.0, floor[-s] - rock_span)
        for s in S.SIDES:
            q[s] = float(np.clip(q[s], 0.0, L.TRAVEL_MAX + Y_BARB * math.sin(L.ROCK_MAX)))
        qc = (q[1] + q[-1]) / 2
        if qc > L.TRAVEL_MAX:
            q = {s: q[s] - (qc - L.TRAVEL_MAX) for s in S.SIDES}
        self.q = q
        open_ = kicked(self.switch, self.dial)
        for s in S.SIDES:
            if open_:
                self.caught[s] = False
            elif q[s] >= L.TRAVEL_SNAP - 1e-9:
                self.caught[s] = True

    # -- the dial and the switch ------------------------------------------------------------------------------------
    def _dial(self, read) -> bool:
        pushed = False
        for tcp, tips, opening in read:
            if opening * rigstate.FULL_GAP > CLOSED_GAP:
                continue
            z = float(tips[2]) - L.TOP - L.DECK_Z
            if not (L.DIAL_HUB_H - 0.001 <= z <= L.DIAL_HUB_H + L.RIDGE_H):
                continue
            dx, dy = float(tips[0]) - L.UX - L.DIAL_X, float(tips[1]) - L.UY - L.DIAL_Y
            rho = math.hypot(dx, dy)
            if PINCH_BAND[0] <= rho < PINCH_BAND[1]:
                # Pinched: the closed pads hold the grip bar PINCH_R out from the axis, so the bar lies along the line
                # from the axis to the pads and the dial turns with the hand (at dial q the bar points at -q, mod pi).
                psi = math.atan2(dy, dx)
                q = -psi + math.pi * round((self.dial + psi) / math.pi)
                if abs(q - self.dial) <= PINCH_SNAP:  # the pads closed on the bar, not beside it
                    self.dial = float(np.clip(q, 0.0, L.DIAL_FULL))
                    pushed = True
                continue
            if not (0.006 <= rho <= L.DIAL_R + 0.006):
                continue
            psi = math.atan2(dy, dx)
            dc = math.atan2(PAD_HALF + L.RIDGE_W / 2, rho)
            for half in (0.0, math.pi):
                e = _wrap(psi - (half - self.dial))
                if 0.0 <= e < dc:
                    self.dial += dc - e
                    pushed = True
                elif -dc < e < 0.0:
                    self.dial -= dc + e
                    pushed = True
            self.dial = float(np.clip(self.dial, 0.0, L.DIAL_FULL))
        return pushed

    def _switch(self, read) -> bool:
        pushed = False
        for tcp, tips, opening in read:
            if opening * rigstate.FULL_GAP > CLOSED_GAP:
                continue
            z = float(tips[2]) - L.TOP - L.DECK_Z
            if not (0.0 < z < L.KNOB[2]):
                continue
            if abs(float(tips[0]) - L.UX - L.SWITCH_X) > L.KNOB[0] / 2 + PAD_HALF_W:
                continue
            py, ky = float(tips[1]) - L.UY, L.SWITCH_Y + self.switch
            reach = PAD_HALF + L.KNOB[1] / 2
            if py < ky and py + reach > ky:
                self.switch = py + reach - L.SWITCH_Y
                pushed = True
            elif py > ky and py - reach < ky:
                self.switch = py - reach - L.SWITCH_Y
                pushed = True
            self.switch = float(np.clip(self.switch, 0.0, L.SWITCH_STOP))
        if not pushed:
            for n in (0.0, L.SWITCH_STOP):
                if abs(self.switch - n) <= L.SWITCH_CAPTURE:
                    self.switch = n
        return pushed

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        self._switch(read)
        if not self._dial(read):
            self.dial = dial_target(self.switch, self.dial)
        self._tray(self._press_depths(read))

    def states(self) -> dict:
        q = (0.0, 0.0, 0.0, 1.0)
        qc = (self.q[1] + self.q[-1]) / 2
        rock = L.rock_for(self.q[1], self.q[-1])
        pawls = {}
        for s in S.SIDES:
            if kicked(self.switch, self.dial):
                pawls[PAWL[s]] = L.PAWL_UPPER
            elif RAMP[0] <= self.q[s] < RAMP[1] and not self.caught[s]:
                pawls[PAWL[s]] = 0.18
            else:
                pawls[PAWL[s]] = 0.0
        return {UNIT: mk.state_from_poses(GAME, (*L.ORIGIN, *q), joints={TRAY: qc, ROCK: rock, DIAL: self.dial,
                                                                            SWITCH: self.switch}, source="offline"),
                CATCHES: mk.state_from_poses(CATCH_PART, (*L.ORIGIN, *q), joints=pawls, source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(UNIT), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, travel mm, rock deg, dial s, switch mm]


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()
    clock_reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()
    clock_reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the joints and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The tray held by both catches, the dial at 50 s or more and running, both grippers 50 mm clear, for 1 s; held
    1 s more. This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline
    stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
            clock_reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
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
    row["source"] = getattr(parts.get(UNIT), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["travel_mm"], d["rock_deg"], d["dial_s"], d["switch_mm"]])
    row["hist"] = list(_HIST)
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 tray latched, dial at 50 s or more and running, hands off; 0.7 latched and set; 0.5 latched; 0.25 one
    catch held at some step; 0.1 the tray pressed 10 mm at some step; 0 otherwise or if a catch was ever forced. The
    score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. The grippers press and push closed (fingertips together) at yaw 0, closing along y: the closed pair is a
# 12 mm (y) by 21 mm (x) fingertip. The dial is wound by pinching its grip bar and turning the wrist (see dial_plan).
CLOSED = 0.0
OPEN = -0.04695
YAW = 0.0
TIP = 0.013
SAFE = 0.130  # travel height of the TCP: pad tips 117 mm up, 29 mm over the dial's grip bar

SW_X = L.UX + L.SWITCH_X
SW_Y = L.UY + L.SWITCH_Y
REACH = PAD_HALF + L.KNOB[1] / 2  # 10 mm: pad centre to knob centre when they touch
Z_SW = L.DECK_Z + 0.004 + TIP  # pad tips 4 mm over the deck, 5 mm under the knob's top
DIAL_C = (L.UX + L.DIAL_X, L.UY + L.DIAL_Y)
AJAR = -0.012  # finger target: pads 24 mm apart, 9 mm clear of each side of the 6 mm grip bar (the tray is down)
Z_DIAL = L.DECK_Z + L.DIAL_HUB_H + 0.004 + TIP  # pad tips 4 mm over the hub, 6 mm under the grip bar's top
Z_DIAL_UP = Z_DIAL + 0.016  # pad tips 10 mm over the grip bar's top: the open pads turn back over it
# The right wrist reaches yaw -70 to +160 deg at the dial (joint 6 limits). At dial angle q the bar lies along yaw -q
# (mod 180 deg), so each grasp turns the wrist clockwise within that window and regrips 180 deg back: three grasps,
# 62 + 180 + 58 = 300 deg, ending at the stop.
GRASPS = [(math.radians(0.0), math.radians(-62.0)),
          (math.radians(118.0), math.radians(-62.0)),
          (math.radians(118.0), math.radians(60.0))]
TURN_STEP = math.radians(45.0)  # at most this much turn per waypoint while pinching (shortest-path rotation)
BACK_STEP = math.radians(60.0)  # and while turning the open hand back over the bar
PRESS_X = L.UX + L.TRAY_XC
PRESS_Y = 0.100  # the pads 100 mm either side of the tray's centre line (TCPs 200 mm apart), 15 mm in from its edge
Z_ABOVE_TRAY = L.TRAY_TOP0 + 0.002 + TIP  # pad tips 2 mm over the popped tray
Z_PRESS = L.DECK_Z - 0.015 + TIP  # pad tips commanded 15 mm below flush: past the snap whatever the arms give
Z_LIFT = L.TRAY_TOP0 + 0.006 + TIP
PARK = {"left": (0.24, 0.27, 0.180), "right": (0.22, -0.27, 0.180)}
WAIT = {"left": (0.30, 0.25, 0.180)}


def _w(d, x, y, z, finger, label, yaw=YAW):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def switch_to(y_from: float, y_to: float, label: str) -> list:
    return [
        _w(1.5, SW_X, y_from, SAFE, CLOSED, f"over the switch, {label}"),
        _w(1.0, SW_X, y_from, Z_SW, CLOSED, "down beside the knob"),
        _w(1.2, SW_X, y_to, Z_SW, CLOSED, f"slide it to {label}"),
        _w(0.5, SW_X, y_to + (0.006 if y_to < y_from else -0.006), Z_SW, CLOSED, "back off the knob"),
        _w(0.8, SW_X, y_to + (0.006 if y_to < y_from else -0.006), SAFE, CLOSED, "up"),
    ]


def pinch_point(yaw: float) -> tuple[float, float]:
    """The TCP when the pads hold the grip bar, turned to ``yaw``: on the bar, PINCH_R out from the dial's axis."""
    return DIAL_C[0] + PINCH_R * math.cos(yaw), DIAL_C[1] + PINCH_R * math.sin(yaw)


def _turn(y0: float, y1: float, step: float, d: float, z: float, finger: float, label: str) -> list:
    n = max(1, math.ceil(abs(y1 - y0) / step - 1e-9))
    out = []
    for i in range(1, n + 1):
        yaw = y0 + (y1 - y0) * i / n
        out.append(_w(d, *pinch_point(yaw), z, finger, f"{label} {i}/{n}", yaw))
    return out


def dial_plan() -> list:
    """Wind the dial 300 deg clockwise to its stop: pads ajar astride the grip bar, close on it, turn the wrist
    clockwise about the dial's axis; let go, lift over the bar, turn back 180 deg, regrip; three grasps."""
    y0 = GRASPS[0][0]
    plan = [_w(1.5, *pinch_point(y0), SAFE, AJAR, "over the dial, pads ajar", y0),
            _w(1.0, *pinch_point(y0), Z_DIAL, AJAR, "down astride the grip bar", y0)]
    prev = None
    for k, (ya, yb) in enumerate(GRASPS, 1):
        if prev is not None:
            plan += _turn(prev, ya, BACK_STEP, 0.6, Z_DIAL_UP, AJAR, f"turn the open hand back for grasp {k}")
            plan.append(_w(0.8, *pinch_point(ya), Z_DIAL, AJAR, "down astride the grip bar", ya))
        plan.append(_w(0.6, *pinch_point(ya), Z_DIAL, CLOSED, f"pinch the grip bar, grasp {k}", ya))
        plan += _turn(ya, yb, TURN_STEP, 0.8, Z_DIAL, CLOSED, f"turn the dial clockwise, grasp {k}")
        plan.append(_w(0.5, *pinch_point(yb), Z_DIAL, AJAR, "let go of the grip bar", yb))
        plan.append(_w(0.6, *pinch_point(yb), Z_DIAL_UP, AJAR, "up off the grip bar", yb))
        prev = yb
    plan.append(_w(0.8, *pinch_point(prev), SAFE, CLOSED, "up off the dial", prev))
    return plan


def press_plan(side: str, t_start: float, t_now: float, variant: str = "") -> list:
    """One arm's part of the two-hand press: wait until t_start, over its side, down to the tray, press to 15 mm
    below flush with the other arm, hold, lift together."""
    y = PRESS_Y if side == "left" else -PRESS_Y
    plan = []
    if t_start > t_now:
        plan.append(_w(t_start - t_now, PRESS_X, y, SAFE, CLOSED, "over the tray, wait for the other hand"))
    z_press = Z_PRESS if variant != "shallow" else L.DECK_Z + 0.006 + TIP
    plan += [
        _w(1.0, PRESS_X, y, Z_ABOVE_TRAY, CLOSED, "down onto the tray's edge"),
        _w(3.0, PRESS_X, y, z_press, CLOSED, "press the tray down level with the other hand"),
        _w(1.0, PRESS_X, y, z_press, CLOSED, "hold at the bottom: both catches click"),
        _w(1.5, PRESS_X, y, Z_LIFT, CLOSED, "lift off together: the catches hold the tray"),
        _w(0.8, PRESS_X, y, SAFE, CLOSED, "up"),
    ]
    return plan


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work (the right arm does the front border, both press): switch to STOP; both hands press the tray down
    level until both catches hold (at STOP the cam is off, so they hold with the dial still at 0); wind the dial
    clockwise to its 60 s stop (the tray is down, clear of the hand turning over the dial); switch to START; both arms
    aside."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = []
    if variant != "no_stop":
        right += switch_to(SW_Y - REACH - 0.008, SW_Y + L.SWITCH_STOP - REACH + 0.001, "STOP")
    right.append(_w(1.5, PRESS_X, -PRESS_Y, SAFE, CLOSED, "over the tray's right edge"))
    t_press = _duration(right)
    right += press_plan("right", t_press, t_press, variant)
    if variant != "no_wind":
        right += dial_plan()
    if variant != "no_start":
        right += switch_to(SW_Y + L.SWITCH_STOP + REACH + 0.008, SW_Y + REACH - 0.001, "START")
    right.append(_w(1.5, *PARK["right"], CLOSED, "aside"))
    left = [_w(2.0, *WAIT["left"], CLOSED, "wait clear of the right arm")]
    left.append(_w(max(0.5, t_press - 1.5 - _duration(left)), *WAIT["left"], CLOSED, "wait"))
    left.append(_w(1.5, PRESS_X, PRESS_Y, SAFE, CLOSED, "over the tray's left edge"))
    left += press_plan("left", t_press, _duration(left), variant)
    left.append(_w(1.5, *PARK["left"], CLOSED, "aside"))
    if variant == "hold_on":
        right = right[:-1] + [_w(1.5, SW_X, SW_Y + REACH, Z_SW, CLOSED, "stay on the switch")]
    return {"left": left, "right": right}


# Wrong ways (counterfactuals), each with the score it earns.
VARIANTS = ("no_stop", "no_wind", "no_start", "shallow", "hold_on")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_stop   the switch left at START: the cam (dial at 0) holds the catches open while the tray is pressed, so
                it pops back up; wound and running, but nothing latched: 0.1
      no_wind   the dial left at 0 (switch to STOP, tray latched, switch to START: the cam opens the pawls and the
                tray pops): 0.25
      no_start  latched and wound but the clock never started: 0.7
      shallow   pressed only to 6 mm above flush: no catch: 0.1
      hold_on   everything done but the right arm stays on the switch: 0.7"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
