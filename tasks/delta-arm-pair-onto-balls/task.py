"""On the station table: one sprung arm pair of a delta 3D printer, its slider and its effector, built from stock
parts to the printer's layout (layout.py has every size and the spec section each comes from).

Three objects:
- ``slider_arms``: a fixed mechkit articulation. Base: the slider plate (50 x 30 x 8 mm) with its stud pair and the
  two keeper pins standing in the bench. Link ``arm_a``: a black tube with a nylon socket block at each end, hinged
  to the base about a vertical axis through the -x slider ball (its far socket swivelling on that ball). Link
  ``arm_b``: the same, mirrored, hinged TO ``arm_a`` about a vertical axis through the +x slider ball, with the hinge
  spring (19.85 N m/rad, the two extension springs' pull at the sockets) and the stop screw as its lower limit. The
  spring's preload is a constant drive target of -0.01905 rad written from an on_step callback every step, because
  mechkit zeroes drive targets at each reset. The two drawn spring struts and the stop screw collide with everything
  but the other arm and carry no force.
- ``rest_bar``: a fixed compound (bar and two steel stop pins) under the arms; the pins stop the pair's sideways
  swing on each tube's outer face.
- ``effector``: one loose rigid body, 139 g: a hexagonal plate, a post to hold it by and three stud pairs, each ball
  three crossed discs (no sphere in the kit).

Goal: hold the pair open, slide the effector's front stud pair between the free sockets, let the springs close the
sockets on the balls, leave the effector flat and both grippers clear. scoring.py has the tiers.

Nothing here applies a force, a weld or a rule to any part; the only per-step write is the spring's preload target.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1 (both arms hold the ready pose).  One arm: DEMO_IDLE_ARM=left (or right).
Variants (counterfactual runs, never the submitted demo): DEMO_VARIANT=<name>, see VARIANTS and demo().
Trace:   TASK_TRACE=<file> writes one JSON line per control step.
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
import tiltgrip  # noqa: E402
from offline_delta import OfflineDelta  # noqa: E402
from scoring import EFFECTOR, REST_BAR, SLIDER_ARMS, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.05  # one control step: the scorer's own 20-step window is the 1 s of the 1.0 line (spec 4.1)

INSTRUCTION = ("Fit the black arm pair onto the blue effector: one socket sprung over each ball of one of its "
               "ball-stud pairs. Leave the effector flat on the table, then let go of both and move the grippers "
               "clear of them.")


# -- shapes ---------------------------------------------------------------------------------------------------------
def _shape(t):
    """A layout.py shape tuple (mm) as a mechkit shape (m)."""
    if t[0] == "box":
        _, c, s, yaw, col = t
        turn = {} if abs(yaw) < 1e-12 else {"ypr": (yaw, 0.0, 0.0)}
        return mk.box(tuple(L.m(c)), tuple(L.m(s)), col, **turn)
    _, c, r, h, axis, yaw, col = t
    turn = {} if abs(yaw) < 1e-12 else {"ypr": (yaw, 0.0, 0.0)}
    return mk.cylinder(tuple(L.m(c)), r * L.MM, h * L.MM, axis, col, **turn)


def _shapes(rows):
    return tuple(_shape(t) for t in rows)


ARM_A_LINK = mk.hinge(L.ARM_A, "Z", tuple(L.m(L.pivot(L.ARM_A))), _shapes(L.arm_shapes(L.ARM_A)),
                      lower=L.ARM_A_LIMITS[0], upper=L.ARM_A_LIMITS[1], mass=L.ARM_MASS[L.ARM_A],
                      damping=L.ARM_A_DAMPING, friction=L.ARM_A_FRICTION, color=L.BLACK)
ARM_B_LINK = mk.hinge(L.ARM_B, "Z", tuple(L.m(L.pivot(L.ARM_B))), _shapes(L.arm_shapes(L.ARM_B)),
                      lower=L.ARM_B_LIMITS[0], upper=L.ARM_B_LIMITS[1], mass=L.ARM_MASS[L.ARM_B],
                      spring=L.ARM_B_SPRING, damping=L.ARM_B_DAMPING, friction=L.ARM_B_FRICTION, parent=L.ARM_A,
                      color=L.BLACK)


class OfflineBox(mk.Assembly):
    """An assembly exactly as mechkit builds and reads it in the simulator. Only tools/offline/check_task.py calls
    the two methods below: offline each object is one small box that no gripper grips (the checker's own carry proxy
    then leaves it alone and its overlap checks do not see the arms riding over the rest bar), and ``offline_state``
    puts the frame back where it was declared; offline_delta.py moves the joints and the effector."""

    def _offline_part(self):
        return np.zeros(3), np.array([0.01, 0.01, 0.004])

    def offline_box(self):
        c, size = self._offline_part()
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ c
        return tuple(float(v) for v in centre), tuple(float(v) for v in size), float(self.yaw)

    def offline_state(self, state):
        c, _ = self._offline_part()
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ c
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


class SliderArms(OfflineBox):
    def _offline_part(self):  # the slider plate
        size, centre = L.SLIDER_PLATE
        return L.m(centre), L.m(size)


class RestBar(OfflineBox):
    def _offline_part(self):  # the bar without its pins
        size, centre = L.REST_BAR
        return L.m(centre), L.m(size)


class Effector(OfflineBox):
    def _offline_part(self):  # the plate's middle box (too wide for any grip)
        return np.array([0.0, 0.0, L.HEX_THICK / 2 * L.MM]), L.m(L.HEX_BOX)


SLIDER_SPEC = SliderArms(name=SLIDER_ARMS, pos=(*L.SLIDER_ORIGIN, L.TOP), color=L.ALUMINIUM, mass=None, yaw=0.0,
                         shapes=_shapes(L.slider_base_shapes()), joints=(ARM_A_LINK, ARM_B_LINK),
                         friction=L.ARM_FRICTION, **mk.TIGHT_FIT)
REST_BAR_SPEC = RestBar(name=REST_BAR, pos=(*L.REST_BAR_ORIGIN, L.TOP), color=L.ACETAL, mass=None, yaw=0.0,
                        shapes=_shapes(L.rest_bar_shapes()), friction=L.REST_BAR_FRICTION, **mk.TIGHT_FIT)
EFFECTOR_SPEC = Effector(name=EFFECTOR, pos=(*L.EFFECTOR_ORIGIN, L.TOP), color=L.BLUE, mass=L.EFFECTOR_MASS, yaw=0.0,
                         shapes=_shapes(L.effector_shapes()), friction=L.EFFECTOR_FRICTION, **mk.TIGHT_FIT)
for _spec in (SLIDER_SPEC, REST_BAR_SPEC, EFFECTOR_SPEC):
    _spec._check()
OBJECTS = [SLIDER_SPEC, REST_BAR_SPEC, EFFECTOR_SPEC]


# -- the spring's preload: a constant drive target, written every step ------------------------------------------------
PRELOAD = {"writes": 0, "last": None}


@mk.on_step
def _preload(states) -> None:
    """In the simulator, at the start of every control step: arm_b's spring pulls toward -0.01905 rad, which puts
    0.378 N m (1.2 N at the sockets) on the stop screw at rest (spec 3.2). The kit zeroes targets at every reset, so
    this is written every step, not once."""
    PRELOAD["last"] = mk.set_targets(SLIDER_SPEC, L.ARM_B, L.ARM_B_TARGET)
    PRELOAD["writes"] += 1


@mk.on_reset
def _preload_reset() -> None:
    mk.set_targets(SLIDER_SPEC, L.ARM_B, L.ARM_B_TARGET)
    OFFLINE.reset()


_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineDelta()  # only stepped without a simulator (see offline_delta.py)


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the parts: mechkit has no environment AND the states handed in were not
    read from one (mechkit marks those ``source="sim"``)."""
    return mk.env() is None and getattr(objects.get(SLIDER_ARMS), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    """Without a simulator nothing moves the joints or the effector, so offline_delta.py moves them from the arms'
    commanded motion; the scorer then sees those states."""
    OFFLINE.step(arms)
    out = dict(objects)
    out[SLIDER_ARMS] = mk.state_from_poses(SLIDER_SPEC, (*SLIDER_SPEC.pos, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
                                           joints=OFFLINE.joints(), source="offline")
    origin, quat = OFFLINE.effector_pose()
    out[EFFECTOR] = mk.state_from_poses(EFFECTOR_SPEC, (*origin, *quat), np.zeros(6), source="offline")
    return out


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _arm_rows(arms) -> dict:
    out = {}
    try:
        for side in rigstate.ARMS:
            a = arms[side]
            out[side] = {"tcp": [round(float(v), 4) for v in a.tcp], "tips": [round(float(v), 4) for v in a.tips],
                         "opening": round(float(a.opening), 3),
                         "pad_gap_mm": round(float(np.clip(a.opening, 0, 1)) * rigstate.FULL_GAP / L.MM, 1)}
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"rigstate_error": "unreadable snapshot"}
    return out


def success(objects) -> bool:
    """On, both grippers clear, flat and still for the scorer's 1 s window. This runs every control step, so it also
    feeds the tracker (and, without a simulator, the offline kinematics that move the parts)."""
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
           "fallback_resets": _TRACKER.resets, **result.details,
           "preload": {"target_rad": L.ARM_B_TARGET, "writes": PRELOAD["writes"]}}
    s = objects.get(SLIDER_ARMS)
    try:
        a = float(s.joints[L.ARM_A])
        row["arm_a_limit_margin_rad"] = round(min(a - L.ARM_A_LIMITS[0], L.ARM_A_LIMITS[1] - a), 4)
        row["joint_vel"] = {k: round(float(v), 4) for k, v in s.joint_vel.items()}
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    row["arms"] = _arm_rows(arms) if arms is not None else {"rigstate_error": rigstate.error()}
    if offline:
        row["offline"] = {"phi": round(OFFLINE.phi, 5), "held_by": OFFLINE.held_by, "seated": OFFLINE.seated,
                          "perched": OFFLINE.perched, "jammed": OFFLINE.jammed}
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 on, both grippers clear, flat and still over the last 1 s; 0.5 on and one gripper clear; 0.25 on; 0
    otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: fixed waypoints from the setup (spec 5.2), nothing read from the simulator. Both grippers close along
# world x and are rolled about world x for the whole demo (tiltgrip.py): left -0.5 rad (tips toward -y), right +0.6
# rad (tips toward +y), so the wrists lean apart. Positions are TCPs (table frame, m; heights over the table); finger
# targets are minus half the commanded pad gap.
def finger(gap_m: float) -> float:
    return -gap_m / 2.0


ENTER = finger(L.LEFT_ENTER_GAP)  # 8 mm between the tubes
OPEN60 = L.RIGHT_OPEN


def spread_finger(width_mm: float, v: float = L.STATION_V) -> float:
    """The finger target that holds ``width_mm`` at a station ``v`` mm from the pivots (layout's give model, plus
    the GPU C4 loss of about 6 mm of pad gap)."""
    return finger((L.pad_gap_for_width(width_mm, v) + L.GPU_GAP_LOSS_MM) * L.MM)


def turned_grip(arm: str, yaw_add: float = 0.0):
    """The arm's tilted grip, turned ``yaw_add`` rad about the WORLD vertical (so a held effector turns flat on the
    table about its post): rz(yaw_add) applied after the roll, not before it."""
    if yaw_add == 0.0:
        return tiltgrip.tilted(L.YAW[arm], L.ROLL[arm])
    return tiltgrip.TiltedYaw(L.YAW[arm] + yaw_add, L.rz(yaw_add) @ tiltgrip.rotation(L.YAW[arm], L.ROLL[arm]))


def _w(arm: str, duration: float, xyz, fing: float, label: str, yaw_add: float = 0.0, trim: bool = True):
    t = (L.TRIM_LEFT if arm == "left" else L.TRIM_RIGHT) if trim else (0.0, 0.0, 0.0)
    pos = (xyz[0] + t[0], xyz[1] + t[1], L.TOP + xyz[2] + t[2])
    return Waypoint(duration, pos, fing, label, turned_grip(arm, yaw_add))


def _hold(plan, seconds: float, label: str = "hold"):
    last = plan[-1]
    return [Waypoint(seconds, last.pos, last.finger, label, last.yaw)]


ST = (*L.LEFT_STATION,)
PICK = (*L.RIGHT_PICK,)
END_Y = L.RIGHT_PICK[1] + L.SLIDE_DY  # -0.078


def left_spread(arm: str = "left", station=ST, spread: float = L.SPREAD_F, open_s: float = 2.5,
                down_from: float = 2.0) -> list:
    """Over the station, down between the tubes with the pads 8 mm apart, open to the spread (6.0 s)."""
    x, y = station
    return [
        _w(arm, down_from, (x, y, 0.080), ENTER, "over the gap between the tubes, pads 8 mm apart"),
        _w(arm, 1.5, (x, y, L.LEFT_Z), ENTER, "down between the tubes"),
        _w(arm, open_s, (x, y, L.LEFT_Z), spread, "open: the fingers' outer faces push the tubes apart"),
    ]


def left_close_leave(arm: str = "left", station=ST, close_to: float = ENTER, leave: bool = True) -> list:
    x, y = station
    out = [_w(arm, L.CLOSE_S, (x, y, L.LEFT_Z), close_to, "close: the tubes follow the fingers in onto the balls")]
    if leave:
        out += [_w(arm, 1.5, (x, y + 0.015, 0.090), close_to, "rise, back toward the slider"),
                _w(arm, 1.5, L.LEFT_AWAY, close_to, "away")]
    return out


def right_pick(arm: str = "right") -> list:
    """Over the post, down either side of it, close on it (5.0 s)."""
    x, y = PICK
    return [
        _w(arm, 2.0, (x, y, 0.090), OPEN60, "over the post, open 60 mm"),
        _w(arm, 1.5, (x, y, L.RIGHT_Z), OPEN60, "down either side of the post"),
        _w(arm, 1.5, (x, y, L.RIGHT_Z), L.RIGHT_GRIP, "close on the post (12 mm command)"),
    ]


def right_slide(arm: str = "right", dy: float = L.SLIDE_DY, seconds: float = 3.0) -> list:
    x, y = PICK
    return [_w(arm, seconds, (x, y + dy, L.RIGHT_Z), L.RIGHT_GRIP, f"slide the effector +y {dy * 1000:.0f} mm")]


def right_leave(arm: str = "right", y_at: float = END_Y, z_at: float = L.RIGHT_Z) -> list:
    x = PICK[0]
    return [
        _w(arm, 1.5, (x, y_at, z_at), OPEN60, "open to 60 mm"),
        _w(arm, 1.5, (x, y_at - 0.015, 0.100), OPEN60, "rise, back from the effector"),
        _w(arm, 1.5, L.RIGHT_AWAY, OPEN60, "away"),
    ]


def _pad(plan, until: float, label: str = "hold"):
    t = sum(w.duration for w in plan)
    return plan + (_hold(plan, until - t, label) if until - t > 1e-9 else [])


def demo_plan(spread: float = L.SPREAD_F, slide: float = L.SLIDE_DY, left_mode: str = "leave",
              right_mode: str = "leave", station=ST, left_close: float = ENTER) -> dict:
    """The demo (spec 5.2) and its one-row changes. left_mode: leave | hold_open (never closes) | stay (closes and
    stays between the tubes). right_mode: leave | keep (never opens) | hover (opens on the post and stays)."""
    left = left_spread(station=station, spread=spread)  # 6.0 s
    right = right_pick()  # 5.0 s
    right = _pad(right, 6.0, "hold the post while the pair opens")
    right += right_slide(dy=slide)  # 9.0 s
    left = _pad(left, 9.5, "hold the pair open")
    if left_mode == "hold_open":
        left = _pad(left, 30.0, "hold the pair open to the end")
    else:
        left += left_close_leave(station=station, close_to=left_close, leave=(left_mode == "leave"))
    right = _pad(right, 12.0, "hold the effector while the sockets close")
    y_end = PICK[1] + slide
    if right_mode == "leave":
        right += [_w("right", 1.5, (PICK[0], y_end, L.RIGHT_Z), OPEN60, "open to 60 mm")]
        right = _pad(right, 13.5, "wait for the left to rise")
        right += [_w("right", 1.5, (PICK[0], y_end - 0.015, 0.100), OPEN60, "rise, back from the effector"),
                  _w("right", 1.5, L.RIGHT_AWAY, OPEN60, "away")]
    elif right_mode == "hover":
        right += [_w("right", 1.5, (PICK[0], y_end, L.RIGHT_Z), OPEN60, "open on the post and stay there")]
    end = max(sum(w.duration for w in p) for p in (left, right)) + 1.5
    return {"left": _pad(left, end, "hold clear"), "right": _pad(right, end, "hold")}


def naive_plan(dy: float = L.SLIDE_DY) -> dict:
    """N1: the right takes the post and pushes the effector 40 mm into the closed pair; the left stays at ready."""
    r = right_pick() + right_slide(dy=dy) + _hold(right_pick() + right_slide(dy=dy), 1.0)
    r += right_leave()
    return {"right": r}


def front_balls_from_tcp(tcp_xy, turn: float) -> tuple[np.ndarray, np.ndarray]:
    """Table xy (m) of the front stud pair's -x and +x balls when the TCP is on the post's centre at ``tcp_xy`` and
    the effector is turned ``turn`` rad about the vertical through it (flat on the table)."""
    post = np.array(L.POST[1][:2])
    out = []
    for b in L.effector_balls_local()[0]:
        rel = L.rz(turn)[:2, :2] @ (b[:2] - post)
        out.append(np.asarray(tcp_xy, dtype=float) + L.m(rel))
    return out[0], out[1]


def twist_tcp(turn: float, hook: bool) -> tuple[float, float]:
    """Where the TCP goes for the turned slide (spec 2.1). Twist-in (N10, N14): the stud pair's midpoint stays on
    the centre line x 0.300 and the turned pair passes between the closed blocks to END_Y. Hook (N12a, N12b): the
    leading ball (the +x one for a positive turn) is on the socket line, its crown 0.5 mm short of the closed
    socket's mouth on the side it leads, the other ball parked behind the tube ends."""
    b1, b2 = front_balls_from_tcp((0.0, 0.0), turn)
    if not hook:
        mid = (b1 + b2) / 2
        return L.SLIDER_ORIGIN[0] - mid[0], END_Y
    lead, sgn = (b2, 1.0) if turn > 0 else (b1, -1.0)
    mouth_x = L.SLIDER_ORIGIN[0] + sgn * L.SOCKET_REST_GAP / 2 * L.MM
    want_x = mouth_x - sgn * (L.BALL_R + 0.5) * L.MM
    return want_x - lead[0], OfflineDelta.socket_y() - lead[1]


TWIST_TURN = math.radians(-45.0)  # N10, N14: the -x ball leads; at +45 deg the right wrist's joint 6 hits its limit


def twist_plan(arm: str = "right", turn: float = TWIST_TURN, shift_x: float = 0.0,
               let_go_and_spread: bool = False) -> dict:
    """N10 (and N12a/N12b with a small turn and a sideways pull, N14 with the spread after): turn the held effector
    about the vertical through its post, slide it in turned (twist_tcp), then turn it square back on x 0.300 (or pull
    it sideways, still turned)."""
    x, y = PICK
    tx, ty = twist_tcp(turn, hook=bool(shift_x))
    p = right_pick(arm)
    p += [_w(arm, 2.0, (x, y, L.RIGHT_Z), L.RIGHT_GRIP, "turn the effector about the post", turn),
          _w(arm, 3.0, (tx, ty, L.RIGHT_Z), L.RIGHT_GRIP, "slide it in turned", turn)]
    if shift_x:
        end_xy = (tx + shift_x, ty)
        p += [_w(arm, 2.0, (*end_xy, L.RIGHT_Z), L.RIGHT_GRIP, "pull it sideways", turn)]
    else:
        end_xy = (x, ty)
        p += [_w(arm, 2.0, (*end_xy, L.RIGHT_Z), L.RIGHT_GRIP, "turn it square: the balls pry the tubes")]
    yaw_end = turn if shift_x else 0.0
    p += [_w(arm, 1.0, (*end_xy, L.RIGHT_Z), L.RIGHT_GRIP, "dwell", yaw_end),
          _w(arm, 1.5, (*end_xy, L.RIGHT_Z), OPEN60, "let go", yaw_end),
          _w(arm, 1.5, (end_xy[0], end_xy[1] - 0.015, 0.100), OPEN60, "rise", yaw_end)]
    if let_go_and_spread:
        p += one_arm_spread(arm)
    else:
        p += [_w(arm, 1.5, L.RIGHT_AWAY, OPEN60, "away")]
    return {arm: p}


def one_arm_spread(arm: str) -> list:
    """The same gripper goes to the station, spreads to W 92, holds, closes and leaves (N13, N14, N17)."""
    x, y = ST
    return [_w(arm, 2.5, (x, y, 0.080), ENTER, "over the gap between the tubes, pads 8 mm apart"),
            _w(arm, 1.5, (x, y, L.LEFT_Z), ENTER, "down between the tubes"),
            _w(arm, 2.5, (x, y, L.LEFT_Z), L.SPREAD_F, "spread to W 92"),
            _w(arm, 1.5, (x, y, L.LEFT_Z), L.SPREAD_F, "hold"),
            _w(arm, L.CLOSE_S, (x, y, L.LEFT_Z), ENTER, "close"),
            _w(arm, 1.5, (x, y + 0.015, 0.090), ENTER, "rise"),
            _w(arm, 2.0, L.LEFT_AWAY if arm == "left" else L.RIGHT_AWAY, ENTER, "away", trim=False)]


def park_plan(arm: str = "right") -> dict:
    """N13: push the effector to the jam, let go, then spread the pair with the same gripper."""
    p = right_pick(arm) + right_slide(arm) + [
        _w(arm, 1.5, (PICK[0], END_Y, L.RIGHT_Z), OPEN60, "let go at the jam"),
        _w(arm, 1.5, (PICK[0], END_Y - 0.015, 0.100), OPEN60, "rise")]
    return {arm: p + one_arm_spread(arm)}


PERCH_LIFT = 0.020  # m: carried level 20 mm above the table (V8 N18)
PERCH_SET = 0.012  # m: front balls at about z 25, on the block tops
PERCH_SHORT = 0.0045  # m: the drop's 3.9 mm plus half the stagger at W 88 (0.6 mm) (V8 N17)
# The balls land midway between the two socket axes, and the plate's slanted front edges clear the closed blocks'
# inboard corners by about 0.8 mm (kit distance on the built shapes; 0.4 mm at 3.7 mm short).


def perch_plan(arm: str = "right", then_spread: bool = True) -> list:
    """N17 / N19: lift the effector by its post, set its front balls on the closed pair's free socket blocks with the
    rear edge of the plate on the table, let go; N17 then spreads the pair with the same gripper."""
    x, y = PICK
    yp = END_Y - PERCH_SHORT
    p = right_pick(arm) + [
        _w(arm, 1.0, (x, y, L.RIGHT_Z + PERCH_LIFT), L.RIGHT_GRIP, "lift 20 mm"),
        _w(arm, 3.0, (x, yp, L.RIGHT_Z + PERCH_LIFT), L.RIGHT_GRIP, "carry over the free socket blocks"),
        _w(arm, 1.5, (x, yp, L.RIGHT_Z + PERCH_SET), L.RIGHT_GRIP, "set the front balls on the block tops"),
        _w(arm, 1.5, (x, yp, L.RIGHT_Z + PERCH_SET), OPEN60, "let go: the rear edge drops to the table"),
        _w(arm, 1.5, (x, yp - 0.020, 0.100), OPEN60, "rise, back from the effector")]
    if then_spread:
        return p + one_arm_spread(arm)
    return p + [_w(arm, 1.5, L.RIGHT_AWAY, OPEN60, "away")]


def hold_at_84_plan() -> list:
    """N18 / N19: the left holds the pair at W 84 to the end."""
    return _pad(left_spread(spread=spread_finger(L.WIDTH_ON)), 40.0, "hold the pair at W 84")


def hover_plan() -> list:
    """N18: the right holds the effector level 20 mm above the table, front balls over the sockets' axes in plan."""
    x, y = PICK
    p = right_pick() + [
        _w("right", 1.0, (x, y, L.RIGHT_Z + PERCH_LIFT), L.RIGHT_GRIP, "lift 20 mm"),
        _w("right", 3.0, (x, END_Y, L.RIGHT_Z + PERCH_LIFT), L.RIGHT_GRIP, "over the sockets, level")]
    return _pad(p, 40.0, "hold it there")


# -- calibration jobs (spec 5.4; one arm each, never the submitted demo; raw: no TRIM applied) ------------------------
def c4_plan() -> dict:
    """C4, left alone: W and the pair's centre line at pad gaps 56, 60 and 64 mm, 2 s each. Design (layout's give
    model): W 86.7, 91.2, 95.8 mm."""
    x, y = ST
    p = [_w("left", 2.0, (x, y, 0.080), ENTER, "over the gap between the tubes, pads 8 mm apart", trim=False),
         _w("left", 1.5, (x, y, L.LEFT_Z), ENTER, "down between the tubes", trim=False)]
    for gap, secs in ((56, 2.5), (60, 1.0), (64, 1.0)):
        p += [_w("left", secs, (x, y, L.LEFT_Z), finger(gap * L.MM), f"open to a {gap} mm pad gap", trim=False),
              _w("left", 2.0, (x, y, L.LEFT_Z), finger(gap * L.MM), f"hold at {gap} mm", trim=False)]
    p += [_w("left", L.CLOSE_S, (x, y, L.LEFT_Z), ENTER, "close to 8 mm", trim=False),
          _w("left", 1.5, (x, y + 0.015, 0.090), ENTER, "rise, back toward the slider", trim=False),
          _w("left", 1.5, L.LEFT_AWAY, ENTER, "away", trim=False)]
    return {"left": p}


C5_SLIDE = 0.025  # m: the closed pair stops the balls 28 mm in (the jam), so one arm reads 25 mm, not 40


def c5_plan() -> dict:
    """C5, right alone: the grasp as in the demo, hold 1 s, slide +y 25 mm in 2 s, hold 2 s, let go. Design: effector
    origin (0.300, -0.083), yaw under 1 deg, plate flat."""
    x, y = PICK
    p = [_w("right", 2.0, (x, y, 0.090), OPEN60, "over the post, open 60 mm", trim=False),
         _w("right", 1.5, (x, y, L.RIGHT_Z), OPEN60, "down either side of the post", trim=False),
         _w("right", 1.5, (x, y, L.RIGHT_Z), L.RIGHT_GRIP, "close on the post (12 mm command)", trim=False),
         _w("right", 1.0, (x, y, L.RIGHT_Z), L.RIGHT_GRIP, "hold the post", trim=False),
          _w("right", 2.0, (x, y + C5_SLIDE, L.RIGHT_Z), L.RIGHT_GRIP, "slide the effector +y 25 mm", trim=False),
          _w("right", 2.0, (x, y + C5_SLIDE, L.RIGHT_Z), L.RIGHT_GRIP, "hold after the slide", trim=False),
          _w("right", 1.5, (x, y + C5_SLIDE, L.RIGHT_Z), OPEN60, "open to 60 mm", trim=False),
          _w("right", 1.5, (x, y + C5_SLIDE - 0.015, 0.100), OPEN60, "rise, back from the effector", trim=False),
          _w("right", 1.5, L.RIGHT_AWAY, OPEN60, "away", trim=False)]
    return {"right": p}


C7_AIR = 5.0  # mm: the finger's outer face starts this far outside the tube's outer face
C7_PUSH = 30.0  # mm the first push travels: 5 of air, the 12.3 mm swing at the station, 12.7 against the stop


def c7_tcp_x() -> tuple[float, float, float, float]:
    """C7's TCP x values (table, m): start and end of the push toward -x from outside the +x tube, then start and end
    of the push toward +x from outside the -x tube as the first push left it (on its pin), with the same 12.7 mm
    asked against the stop. The closed pads (8 mm) and fingers (8 mm) reach 12 mm from the TCP."""
    reach = L.LEFT_ENTER_GAP / L.MM / 2 + L.FINGER_THICK
    face = L.arm_point(L.ARM_B, L.TUBE_OUTER_U, L.STATION_V, 0.0)[0]  # +x tube's outer face, slider frame (mm)
    swing = L.STATION_V * L.pin_stop_angle()  # 12.3 mm at the station
    x1 = face + C7_AIR + reach
    x2 = x1 - C7_PUSH
    x3 = -(face + swing) - C7_AIR - reach  # the -x tube's outer face, swung onto its pin
    x4 = x3 + C7_AIR + 2 * swing + (C7_PUSH - C7_AIR - swing)
    return tuple(L.SLIDER_ORIGIN[0] + v * L.MM for v in (x1, x2, x3, x4))


def c7_plan() -> dict:
    """C7, left alone, pads 8 mm apart: push the closed pair sideways by a tube's outer face, each way, 3 s and a 2 s
    hold. Design: arm_a at -0.050 then +0.050 rad, W 74.0 throughout, arm_a never within 0.02 rad of +-0.15, the
    pair stays where it is left."""
    y = L.LEFT_STATION[1]
    x1, x2, x3, x4 = c7_tcp_x()
    p = [_w("left", 2.5, (x1, y, 0.080), ENTER, "over the +x tube's outer side, pads 8 mm apart", trim=False),
         _w("left", 1.5, (x1, y, L.LEFT_Z), ENTER, "down outside the +x tube", trim=False),
         _w("left", 3.0, (x2, y, L.LEFT_Z), ENTER, "push the pair toward -x onto arm_a's pin", trim=False),
         _w("left", 2.0, (x2, y, L.LEFT_Z), ENTER, "hold against the pin", trim=False),
         _w("left", 1.0, (x2 + 0.015, y, L.LEFT_Z), ENTER, "back out", trim=False),
         _w("left", 1.0, (x2 + 0.015, y, 0.080), ENTER, "rise", trim=False),
         _w("left", 2.5, (x3, y, 0.080), ENTER, "cross over the pair to the -x side", trim=False),
         _w("left", 1.5, (x3, y, L.LEFT_Z), ENTER, "down outside the -x tube", trim=False),
         _w("left", 3.0, (x4, y, L.LEFT_Z), ENTER, "push the pair toward +x onto arm_b's pin", trim=False),
         _w("left", 2.0, (x4, y, L.LEFT_Z), ENTER, "hold against the pin", trim=False),
         _w("left", 1.0, (x4 - 0.015, y, L.LEFT_Z), ENTER, "back out", trim=False),
         _w("left", 1.0, (x4 - 0.015, y, 0.080), ENTER, "rise", trim=False),
         _w("left", 1.5, L.LEFT_AWAY, ENTER, "away", trim=False)]
    return {"left": p}


N17_ARM = "right"  # the arm that solves every N17 waypoint (tools/offline/check_task.py with DEMO_VARIANT=N17)

VARIANTS = ("N0", "N1", "N2", "N3", "N4", "N5_86", "N5_83", "N5_80", "N6", "N7", "N8", "N8b", "N9", "N10", "N12a",
            "N12b", "N13", "N14", "N15", "N16", "N17", "N18", "N19", "C4", "C5", "C7")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual run (spec 2.2, V8)."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "N0":
        return {}
    if variant in ("N1", "N3"):
        return naive_plan() if variant == "N1" else {"right": demo_plan()["right"]}
    if variant == "N2":
        return {"left": demo_plan()["left"]}
    if variant == "N4":
        return demo_plan(spread=spread_finger(98.0))
    if variant.startswith("N5_"):
        return demo_plan(spread=spread_finger(float(variant[3:])))
    if variant == "N6":
        return demo_plan(left_mode="hold_open")
    if variant == "N7":
        return demo_plan(left_mode="stay", left_close=finger(0.040), right_mode="keep")
    if variant == "N8":
        return demo_plan(right_mode="keep")
    if variant == "N8b":
        return demo_plan(left_mode="stay")
    if variant == "N9":
        return demo_plan(slide=L.SLIDE_DY - 0.008)
    if variant == "N10":  # turned -45 deg: +45 pins the right wrist's last joint (offline check)
        return twist_plan(turn=TWIST_TURN)
    if variant == "N12a":  # the +x ball (arm_b, the child) hooked, pulled +x
        return twist_plan(turn=math.radians(16.0), shift_x=0.020)
    if variant == "N12b":  # the -x ball (arm_a, the parent) hooked, pushed -x
        return twist_plan(turn=math.radians(-16.0), shift_x=-0.020)
    if variant == "N13":
        return park_plan()
    if variant == "N14":
        return twist_plan(turn=TWIST_TURN, let_go_and_spread=True)
    if variant == "N15":
        return demo_plan(right_mode="hover")
    if variant == "N16":
        st = (L.LEFT_STATION[0], 0.130)
        return demo_plan(station=st, spread=spread_finger(L.HOLD_WIDTH, (L.SLIDER_ORIGIN[1] - st[1]) / L.MM))
    if variant == "N17":
        return {N17_ARM: perch_plan(N17_ARM)}
    if variant == "N18":
        return {"left": hold_at_84_plan(), "right": hover_plan()}
    if variant == "N19":
        return {"left": hold_at_84_plan(), "right": perch_plan("right", then_spread=False)}
    if variant in ("C4", "C5", "C7"):
        return {"C4": c4_plan, "C5": c5_plan, "C7": c7_plan}[variant]()
    return demo_plan()


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, the settle job).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm checks).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo; spec 2.2 and overrides V8):
      N0 nothing; N1 naive push (right only); N2 / N3 the demo with the right / left arm idle; N4 spread to W 98;
      N5_86, N5_83, N5_80 spread only to that width; N6 the left never closes; N7 the left closes to a 40 mm pad gap
      and stays, the right never opens; N8 the right never opens; N8b the left closes and stays between the tubes;
      N9 the slide stops 8 mm short; N10 twist-in (right only); N12a / N12b hook, swing and stretch on the child /
      parent side (right only); N13 park, then spread (right only); N14 prop, then spread (right only); N15 the right
      opens on the post and stays; N16 the spread at 145 mm from the slider's balls; N17 perch, then spread (one
      arm); N18 the left holds W 84, the right holds the effector level 20 mm up; N19 the left holds W 84, the right
      sets the front balls on the block tops and leaves.
    Calibration jobs (spec 5.4, one arm each): C4 the left opens the pair to pad gaps of 56, 60 and 64 mm, 2 s each;
      C5 the right takes the post and slides the effector +y 25 mm; C7 the left pushes the closed pair sideways onto
      each rest-bar pin by a tube's outer face."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
