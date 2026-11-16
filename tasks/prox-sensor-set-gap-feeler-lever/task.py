"""On the station table, real size: a Turck BI2-M12-AP6X-H1141 inductive sensor (M12 x 1, 52 mm, 31 g, Sn 2 mm) lies in
an aluminium split clamp block (original part, 22 x 45 x 25 mm) with its face 12.0 mm short of a steel trip dog at its
end stop. The block's KIPP K0006.1501406X20 cam lever (A 70.4 mm, red) stands up open. An 8 in (203 mm) 1.00 mm feeler
leaf, 12.7 mm wide, in a 22 mm handle, lies on its edge in a rack tray 80 mm short of the gap and 230 mm to the right.

Goal: the sensing gap set to 1.00 mm (half of Sn) and locked, the leaf out and back in its tray. The gap is set by
contact: the leaf is slid in from the side, with one hand holding it upright, until it stands between the sensor's face
and the dog; the other hand pushes the sensor up to it (the leaf is pinched and pushed against the dog); the lever is
shut while the leaf still stands in the gap; only then does the leaf come out. The leaf's tail is 200 mm long, so the
two hands work 20 cm apart.

The rig (clamp, dog, sensor slide, lever) is one mechkit mechanism, the leaf and the tray are mechkit compounds
(mechkit.py, next to this file). Two declared software rules (``_rig_rule``, a mechkit on_step callback): (1) the
sensor's slide is a position drive that follows the sensor each step while the lever is open (a light drag), and holds
the position it had when the lever reached its last 5 degrees once shut (the clamp: 2 N per mm); (2) the lever has two
notches (open, shut) that the drive snaps to within 0.35 rad (the cam's over-centre detent). The scorer reads the
slide's and the lever's joints and the leaf's pose. The arms are read through rigstate.py (the let-go check).

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
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import LEAF, RIG  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Set the proximity sensor's gap to 1.0 mm with the feeler leaf, lock the clamp lever, and put the leaf "
               "back in its tray.")


# -- the parts --------------------------------------------------------------------------------------------------------
def dog_shapes():
    out = [mk.box((L.X_DOG + L.DOG_T / 2, 0.0, (L.DOG_Z0 + L.DOG_Z1) / 2), (L.DOG_T, L.DOG_Y, L.DOG_Z1 - L.DOG_Z0),
                  L.DOG_STEEL),
           mk.box(((L.X_DOG + L.DOG_T + L.CAR_X1) / 2, 0.0, L.CAR_H / 2),
                  (L.CAR_X1 - L.X_DOG - L.DOG_T, L.CAR_W, L.CAR_H), L.ALU)]
    return out


def clamp_shapes():
    """The split block (22 along the bore, 45 across, 25 tall) round a 12.4 mm square bore on the sensor's axis."""
    xc, w = (L.CL_X0 + L.CL_X1) / 2, L.CL_X1 - L.CL_X0
    hb, z0, z1, y = L.BORE / 2, L.AXIS_Z - L.BORE / 2, L.AXIS_Z + L.BORE / 2, L.CL_Y / 2
    return [mk.box((xc, 0.0, z0 / 2), (w, L.CL_Y, z0), L.BLOCK_ALU),
            mk.box((xc, 0.0, (z1 + L.CL_H) / 2), (w, L.CL_Y, L.CL_H - z1), L.BLOCK_ALU),
            mk.box((xc, (hb + y) / 2, (z0 + z1) / 2), (w, y - hb, z1 - z0), L.BLOCK_ALU),
            mk.box((xc, -(hb + y) / 2, (z0 + z1) / 2), (w, y - hb, z1 - z0), L.BLOCK_ALU)]


def sensor_shapes():
    barrel_l = L.S_LEN - L.CONN_L
    return [mk.cylinder((L.S_FACE0 - barrel_l / 2, 0.0, L.AXIS_Z), L.S_R, barrel_l, "X", L.BRASS),
            mk.cylinder((L.S_FACE0 - 0.001, 0.0, L.AXIS_Z), L.S_R, 0.002, "X", L.SENSOR_YELLOW),
            mk.cylinder((L.S_FACE0 - barrel_l - L.CONN_L / 2, 0.0, L.AXIS_Z), L.CONN_R, L.CONN_L, "X", L.CONN_BLACK)]


def lever_shapes():
    px, py, pz = L.LEVER_PIVOT
    e = L.LEVER_OPEN
    c = (px, py - L.LEVER_L / 2 * math.cos(e), pz + L.LEVER_L / 2 * math.sin(e))
    return [mk.box(c, (L.LEVER_W, L.LEVER_L, L.LEVER_T), L.LEVER_RED, ypr=(0.0, 0.0, -e))]


SENSOR = mk.slider("sensor", "X", sensor_shapes(), upper=L.SLIDE_UP, lower=L.SLIDE_LO, mass=L.S_MASS, spring=L.SLIDE_K,
                   damping=L.SLIDE_DAMP)
LEVER = mk.hinge("lever", "X", L.LEVER_PIVOT, lever_shapes(), upper=L.LEVER_MAX, lower=0.0, mass=L.LEVER_MASS,
                 spring=L.LEVER_SPRING, damping=L.LEVER_DAMP, friction=L.LEVER_FRICTION)
RIG_PART = mk.assembly(RIG, origin=(0.0, 0.0, L.TOP), shapes=dog_shapes() + clamp_shapes(), joints=[SENSOR, LEVER],
                       color=L.BLOCK_ALU, friction=(0.6, 0.5), tight_fit=True)


def leaf_shapes():
    return [mk.box((0.0, -L.LF_L / 2, L.LF_W / 2), (L.LF_T, L.LF_L, L.LF_W), L.LF_STEEL),
            mk.box((0.0, L.LF_H_YC, L.LF_H_Z / 2), (L.LF_H_X, L.LF_H_Y, L.LF_H_Z), L.LF_ORANGE)]


TRAY_PART = mk.assembly("tray", origin=(0.0, 0.0, L.TOP),
                        shapes=mk.open_box((L.TRAY_X, L.TRAY_Y, L.TRAY_OUT[2] / 2), L.TRAY_OUT, L.TRAY_WALL,
                                           L.TRAY_FLOOR, color=L.TRAY_GREEN),
                        color=L.TRAY_GREEN, friction=(0.6, 0.5), tight_fit=True)
LEAF_PART = mk.assembly(LEAF, origin=(L.LEAF_X0, L.LEAF_Y0, L.TOP + L.LEAF_Z0), mass=L.LF_MASS, shapes=leaf_shapes(),
                        color=L.LF_STEEL, friction=L.LF_FRICTION, tight_fit=True)
OBJECTS = [RIG_PART, TRAY_PART, LEAF_PART]


# -- the two declared rules --------------------------------------------------------------------------------------------
_LOCK = {"q": None}  # the slide position held while the lever is shut


def drive_targets(qs: float, ql: float, lock):
    """(sensor target m, lever target rad, lock) from the joints. Open: the sensor's target follows it (drag); shut: it
    holds the position it had when the lever first came within 5 degrees of its stop. The lever snaps to the nearer
    notch (open 0, shut LEVER_MAX) within LEVER_CAPTURE, else holds where it is."""
    closed = ql >= L.LEVER_MAX - L.LEVER_CLOSED_BAND
    if closed and lock is None:
        lock = qs
    elif not closed:
        lock = None
    ts = lock if lock is not None else qs
    ts = min(max(ts, L.SLIDE_LO), L.SLIDE_UP)
    tl = mk.detent_target(ql, (0.0, L.LEVER_MAX), L.LEVER_CAPTURE)
    return ts, tl, lock


@mk.on_step
def _rig_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    rig = states.get(RIG)
    try:
        qs, ql = float(rig.joints["sensor"]), float(rig.joints["lever"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    if not (math.isfinite(qs) and math.isfinite(ql)):
        return
    ts, tl, _LOCK["q"] = drive_targets(qs, ql, _LOCK["q"])
    mk.set_targets(RIG_PART, "sensor", ts)
    mk.set_targets(RIG_PART, "lever", tl)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows nothing of the sensor's slide, the lever or a leaf pushed against the dog. Offline only
# (never in the simulator) this model gives the scorer: (1) the lever turned down by a closed gripper whose pad tips are
# on its tip's top; (2) the sensor slid by a closed gripper whose block touches its rear, stopped by the dog (the leaf
# in the gap is pushed to the dog and pinched), and held where it was once the lever is shut; (3) the leaf carried by a
# closed gripper at its handle, standing on the table (or the tray's floor) where it is let go.
PAD_FACE = 0.005  # a closed pad block's face ahead of the TCP along the closing direction, less the arm's trail


LEVER_CONTACT_S = 0.045  # the pad pushes the lever's inboard face 45 mm up from the pivot (25 mm short of its tip)
RIGHT_TRAIL_Y = 0.031  # the right arm's measured shortfall toward +y at the lever (GPU 2026-11-16: -71 asked, -101 got)


def lever_q_pushed(ty: float, tz: float, q0: float) -> float:
    """The lever angle after a closed pad block (footprint ``ty`` +- the pad's half width across, its bottom at ``tz``
    over the table) has pushed it from ``q0``: the block carries the face point it touches outward and down until
    that point leaves the block (past its outboard edge or under its bottom). A block that does not cover the point
    leaves the lever where it was."""
    hw = rigstate.FINGER_HALF_WIDTH
    q = q0
    while q < L.LEVER_MAX:
        y, z = L.lever_face_point(q, LEVER_CONTACT_S)
        if not (ty - hw <= y <= ty + hw and z >= tz):
            break
        q += 0.002
    return min(q, L.LEVER_MAX)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.q_s = 0.0
        self.q_l = 0.0
        self.lock = None
        self.leaf = np.array([L.LEAF_X0, L.LEAF_Y0, L.LEAF_Z0])  # the tip's bottom centre, z above the table
        self.grip: dict = {}
        self.was_held = False

    def leaf_in_gap(self) -> bool:
        """The blade between the sensor's face and the dog: its tip past the face's near edge, near the gap's x."""
        return bool(-0.004 <= self.leaf[1] <= 0.04 and abs(self.leaf[0] - L.GAP_X) <= 0.012 and self.leaf[2] <= 0.03)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        # 1. the lever
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            tz = float(tips[2]) - L.TOP
            ty, tx = float(tcp[1]), float(tcp[0])
            if side == "right":
                ty -= RIGHT_TRAIL_Y  # the real arm lands this far short of a command at the lever
            if opening > 0.3 or abs(tx - L.LEVER_PIVOT[0]) > 0.02:
                continue
            self.q_l = max(self.q_l, lever_q_pushed(ty, tz, self.q_l))
        closed = self.q_l >= L.LEVER_MAX - L.LEVER_CLOSED_BAND
        if closed and self.lock is None:
            self.lock = self.q_s
        # 2. the sensor (and the leaf it pushes)
        if self.lock is None:
            for tcp, tips, opening in read:
                tz = float(tips[2]) - L.TOP
                if opening > 0.3 or abs(float(tcp[1])) > 0.0135 + 0.007 or not (0.002 <= tz <= 0.03):
                    continue
                face = float(tcp[0]) + PAD_FACE
                rear = L.S_REAR0 + self.q_s
                if face > rear and float(tcp[0]) < rear + 0.010:
                    q_max = L.SLIDE_UP
                    if self.leaf_in_gap():
                        q_max = L.X_DOG - L.LF_T - L.S_FACE0
                    self.q_s = max(self.q_s, min(face - L.S_REAR0, q_max))
        else:
            self.q_s = self.lock
        if self.leaf_in_gap():
            x_new = min(max(self.leaf[0], L.S_FACE0 + self.q_s + L.LF_T / 2), L.X_DOG - L.LF_T / 2)
            dx = x_new - self.leaf[0]
            self.leaf[0] = x_new
            for off in self.grip.values():
                off[0] += dx
        # 3. the leaf
        holder = None
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            tz = float(tips[2]) - L.TOP
            hy = self.leaf[1] + L.LF_H_YC
            holding = (opening < 0.6 and abs(float(tcp[0]) - self.leaf[0]) < 0.012 and abs(float(tcp[1]) - hy) < 0.02
                       and self.leaf[2] - 0.004 < tz < self.leaf[2] + L.LF_H_Z + 0.004)
            if holding:
                holder = side
                if side not in self.grip:
                    self.grip[side] = self.leaf - np.array([float(tcp[0]), float(tcp[1]), float(tcp[2]) - L.TOP])
                self.leaf = np.array([float(tcp[0]), float(tcp[1]), float(tcp[2]) - L.TOP]) + self.grip[side]
            else:
                self.grip.pop(side, None)
        if holder is not None:
            self.was_held = True
        elif self.was_held:
            self._let_go()

    def _let_go(self) -> None:
        """Let go: the leaf stands on the tray's floor if over the tray, else on the table, where it is."""
        self.was_held = False
        x, y = float(self.leaf[0]), float(self.leaf[1])
        hy = y + L.LF_H_YC
        over_tray = abs(x - L.TRAY_X) <= L.TRAY_OUT[0] / 2 and abs(hy - L.TRAY_Y) <= L.TRAY_OUT[1] / 2
        self.leaf = np.array([x, y, L.LEAF_Z0 if over_tray else 0.0002])

    def states(self) -> dict:
        rig = mk.state_from_poses(RIG_PART, (0.0, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                  joints={"sensor": self.q_s, "lever": self.q_l}, source="offline")
        leaf = mk.state_from_poses(LEAF_PART, (self.leaf[0], self.leaf[1], L.TOP + self.leaf[2], 0.0, 0.0, 0.0, 1.0),
                                   source="offline")
        return {RIG: rig, LEAF: leaf}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(RIG), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, gap mm, lever deg, leaf x mm, leaf z mm]


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()
    _LOCK["q"] = None


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()
    _LOCK["q"] = None


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the gap, lever, leaf and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The lever shut on the leaf, the gap 1.00 +- 0.25 mm, the leaf then out and in its tray, both hands 30 mm clear
    and every part still for 1 s, no latched crash; held 1 s. This runs first every control step, so it also feeds the
    tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, "lock_q": _LOCK["q"], **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(RIG), "source", None)
    d = result.details
    if d.get("valid_state") and _TRACKER.steps % 2 == 0:
        _HIST.append([_TRACKER.steps, d["gap_mm"], d["lever_deg"], d["leaf_mm"][0], d["leaf_mm"][2]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the clamp was shut with the face within 0.30 mm of the dog (latched), or unless every part is let go and
    still; 1.0 the lever shut on the leaf standing in the gap, the gap within 0.25 mm of 1.00, the leaf then out of the
    gap and in its tray; 0.75 the same with the leaf elsewhere; 0.5 the gap right with the leaf left in it or the lever
    not shut on it, or shut within 0.50 to 1.62 mm; 0.25 shut
    within 0.30 to 4.0 mm, or the gap right with the lever open; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than a gripped part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
OPEN_R = -(L.LF_H_X + 0.040) / 2  # 62 mm round the 22 mm handle
GRIP_R = -(L.LF_H_X - 0.008) / 2  # 14 mm commanded across the handle
CLOSED = 0.0
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # both close along x

# the right arm: the leaf (handle gripped along x, so it stands at the handle's centre: tip y minus 198 mm)
HANDLE_TIPS = 0.012  # pad tips 12 mm over the leaf's bottom: on the handle's upper 12 of its 24 mm
LIFT = 0.014  # carried 14 mm up: the blade's bottom clears the tray's rim and the table
HOVER_R = 0.100
PARK_R = (0.14, -0.20, 0.150)
TABLE_DROP = (0.370, -0.115)  # variant: the leaf's tip over bare table, out of the tray


def grip_tcp(base_z: float) -> float:
    return base_z + HANDLE_TIPS + PAD_TIP


GRIP_TRAY = grip_tcp(L.LEAF_Z0)
GRIP_TABLE = grip_tcp(0.0003)
HANDLE_Y_TRAY = L.LEAF_Y0 + L.LF_H_YC  # -0.313
HANDLE_Y_GAP = L.GAP_AIM_TIP_Y + L.LF_H_YC  # -0.186
PULL_OUT = 0.060  # the tip pulled this far back from the gap's y

# the left arm: the push
PUSH_TIPS = 0.010  # pad tips 10 mm over the table: on the connector's upper half
PUSH_TCP = PUSH_TIPS + PAD_TIP
PUSH_START_X = L.S_REAR0 - PAD_FACE - 0.012  # the closed block comes down 12 mm short of the connector (in the air)
OVERSHOOT = 0.004  # commanded this far past where the dog stops the sensor (about 2 N at the arm's trail)
PUSH_SPEED = 0.03  # m/s
BACK_OFF = 0.045
PARK_L = (0.22, 0.30, 0.200)
HOVER_L = 0.150

LEVER_X = L.LEVER_PIVOT[0]
LEVER_STEPS_DEG = (70.0, 50.0, 32.0, 15.0, 0.0)  # the elevations the block's corner follows the lever's face through
LEVER_PUSH = 0.004  # the pad's outboard bottom corner is commanded this far into the face (along its normal)
LEVER_STEP_S = 1.0  # seconds per step of the press
LEVER_CLEAR = 0.025  # the block comes down this far inboard of the face, then sweeps out onto it
LEVER_DY = RIGHT_TRAIL_Y  # every right-arm command at the lever is this far further +y (the arm lands this short)
LEVER_LEAD = 2.0  # the right hand's trip round behind the sensor starts this long before the left hand has parked
LEVER_PRESS = 0.008  # the last press is commanded this far under the shut lever's top (the lever's stop holds it)


def push_x(stroke: float, overshoot: float = OVERSHOOT) -> float:
    """The TCP x that has pushed the sensor ``stroke`` m (the face past its start) plus the overshoot."""
    return L.S_REAR0 + stroke + overshoot - PAD_FACE


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def lever_corner(elev_deg: float, inboard: float = 0.0, push: float = LEVER_PUSH):
    """(y, z above the table) of the pad block's outboard bottom corner for a press at lever elevation ``elev_deg``:
    on the inboard face, ``LEVER_CONTACT_S`` from the pivot, ``push`` into it along the face's normal, or (``inboard``)
    that far inboard of it."""
    e = math.radians(elev_deg)
    y, z = L.lever_face_point(L.LEVER_OPEN - e, LEVER_CONTACT_S)
    return y - push * math.sin(e) + inboard, z - push * math.cos(e)


def lever_tip_target(elev_deg: float, inboard: float = 0.0, push: float = LEVER_PUSH):
    """The pad block's TCP (y, z above the table) for a press at ``elev_deg``: the block extends inboard of its outboard
    corner by the pad's width, its bottom 13 mm under the TCP."""
    y, z = lever_corner(elev_deg, inboard, push)
    return y + rigstate.FINGER_HALF_WIDTH, z + PAD_TIP


def leaf_to_gap_moves():
    """The right arm: open over the leaf's handle in its tray, down, grip, lift, carry across to the gap's x, slide the
    blade in from the side until its tip is past the face, lower it to the table."""
    x, y = L.LEAF_X0, HANDLE_Y_TRAY
    ax = L.GAP_AIM_X
    return [
        _w(2.0, x, y, HOVER_R, OPEN_R, "over the leaf's handle in its tray", YAW_R),
        _w(1.0, x, y, GRIP_TRAY + 0.030, OPEN_R, "down to just above the handle", YAW_R),
        _w(1.5, x, y, GRIP_TRAY, OPEN_R, "slowly down round the handle", YAW_R),
        _w(0.5, x, y, GRIP_TRAY, OPEN_R, "settle", YAW_R),
        _w(1.5, x, y, GRIP_TRAY, GRIP_R, "grip the handle", YAW_R),
        _w(1.0, x, y, GRIP_TRAY + LIFT, GRIP_R, "lift the leaf 14 mm", YAW_R),
        _w(2.5, ax, y, GRIP_TRAY + LIFT, GRIP_R, "carry it across to the gap's line", YAW_R),
        _w(5.0, ax, HANDLE_Y_GAP, GRIP_TRAY + LIFT, GRIP_R, "slide the blade in from the side, tip past face", YAW_R),
        _w(0.5, ax, HANDLE_Y_GAP, GRIP_TRAY + LIFT, GRIP_R, "settle", YAW_R),
        _w(1.0, ax, HANDLE_Y_GAP, GRIP_TABLE, GRIP_R, "lower the leaf onto the table, upright", YAW_R),
        _w(0.5, ax, HANDLE_Y_GAP, GRIP_TABLE, GRIP_R, "settle in the gap", YAW_R),
    ]


def lever_moves():
    """The right arm, fingers closed into a block: from the wait beside the leaf round behind the sensor's rear (the
    first GPU run stalled over the open lever's tip, 30 mm short in y, so the hand no longer crosses the lever), in over
    the clamp inboard of the open lever's face, then the block's outboard bottom corner sweeps out onto the face and
    pushes it out and down through the arc to shut (the block pushes the lever, so it cannot run ahead of it), and a
    last press on its top. Commands are LEVER_DY further +y than the places meant (the arm's measured shortfall)."""
    y_in, z_in = lever_tip_target(LEVER_STEPS_DEG[0], inboard=LEVER_CLEAR)
    z_wait = GRIP_TABLE + 0.050  # the height of the wait before it
    dy = LEVER_DY
    plan = [
        _w(2.0, 0.205, -0.12, z_wait, CLOSED, "round behind the sensor's rear, clear of the open lever", YAW_R),
        _w(1.2, 0.205, y_in + dy, z_in, CLOSED, "across, 25 mm inboard of the open lever's face", YAW_R),
        _w(1.5, LEVER_X, y_in + dy, z_in, CLOSED, "in over the clamp, inboard of the lever's face", YAW_R),
    ]
    for e in LEVER_STEPS_DEG:
        y, z = lever_tip_target(e)
        plan.append(_w(LEVER_STEP_S, LEVER_X, y + dy, z, CLOSED, f"push the lever's face down to {e:.0f} degrees",
                       YAW_R))
    plan.append(_w(1.0, LEVER_X, y + dy, z - LEVER_PRESS, CLOSED, "press the shut lever down", YAW_R))
    plan.append(_w(0.5, LEVER_X, y + dy, z - LEVER_PRESS, CLOSED, "hold it shut", YAW_R))
    plan.append(_w(1.0, LEVER_X, y + dy, z + 0.050, CLOSED, "straight up", YAW_R))
    return plan


def leaf_out_moves(drop=None):
    """The right arm: open over the handle of the leaf standing in the gap, down, grip, pull it out sideways, lift,
    carry to the tray (or ``drop``, the tip's x, y), lower, let go."""
    ax = L.GAP_AIM_X
    if drop is None:
        tx, ty, z_rel = L.LEAF_X0, HANDLE_Y_TRAY, GRIP_TRAY + 0.003
    else:
        tx, ty, z_rel = drop[0], drop[1] + L.LF_H_YC, GRIP_TABLE + 0.003
    return [
        _w(2.0, ax, HANDLE_Y_GAP, GRIP_TABLE + 0.040, OPEN_R, "over the leaf's handle at the gap", YAW_R),
        _w(1.5, ax, HANDLE_Y_GAP, GRIP_TABLE, OPEN_R, "slowly down round the handle", YAW_R),
        _w(0.5, ax, HANDLE_Y_GAP, GRIP_TABLE, OPEN_R, "settle", YAW_R),
        _w(1.5, ax, HANDLE_Y_GAP, GRIP_TABLE, GRIP_R, "grip the handle", YAW_R),
        _w(1.0, ax, HANDLE_Y_GAP, GRIP_TABLE + 0.003, GRIP_R, "lift 3 mm", YAW_R),
        _w(3.0, ax, HANDLE_Y_GAP - PULL_OUT, GRIP_TABLE + 0.003, GRIP_R, "pull the blade out sideways", YAW_R),
        _w(1.0, ax, HANDLE_Y_GAP - PULL_OUT, GRIP_TRAY + LIFT, GRIP_R, "lift it clear", YAW_R),
        _w(3.5, tx, ty, GRIP_TRAY + LIFT, GRIP_R, "carry it over the tray", YAW_R),
        _w(1.5, tx, ty, z_rel, GRIP_R, "lower it to just over the floor", YAW_R),
        _w(0.5, tx, ty, z_rel, GRIP_R, "settle", YAW_R),
        _w(1.5, tx, ty, z_rel, OPEN_R, "let go", YAW_R),
        _w(1.0, tx, ty, HOVER_R, OPEN, "straight up", YAW_R),
        _w(1.5, *PARK_R[:2], PARK_R[2], OPEN, "aside", YAW_R),
    ]


def left_approach():
    return [
        _w(3.0, PUSH_START_X, 0.0, HOVER_L, OPEN, "over the air behind the sensor", YAW_L),
        _w(1.5, PUSH_START_X, 0.0, HOVER_L, CLOSED, "close the fingers into a block", YAW_L),
        _w(1.5, PUSH_START_X, 0.0, PUSH_TCP + 0.030, CLOSED, "down to just above the connector's height", YAW_L),
        _w(1.5, PUSH_START_X, 0.0, PUSH_TCP, CLOSED, "slowly down: pad tips 10 mm over the table", YAW_L),
    ]


def left_retreat(x1):
    return [
        _w(1.0, x1 - BACK_OFF, 0.0, PUSH_TCP, CLOSED, "back off 45 mm", YAW_L),
        _w(1.0, x1 - BACK_OFF, 0.0, HOVER_L, CLOSED, "straight up", YAW_L),
        _w(1.5, *PARK_L[:2], PARK_L[2], OPEN, "aside", YAW_L),
    ]


# Wrong ways (counterfactuals): see plans().
VARIANTS = ("short", "crash", "leaf_stays", "leaf_table")
SHORT_STROKE = 0.0090  # the sensor pushed only 9.0 mm: the face stops 3.0 mm from the dog
FULL = L.GAP0 - L.LF_T  # the stroke that pinches the leaf against the dog: 11.0 mm


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the right arm brings the leaf and slides it into the gap, 20 cm from where the left hand will be; the
    left arm pushes the sensor up to it (it pushes the leaf to the dog and pinches it); the right arm lets go and clears
    the gap; the left backs off and parks (the sensor stays: the open clamp drags it lightly); the right shuts the lever
    with a closed block; then takes the leaf out sideways and puts it in the tray."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    stroke = {"short": SHORT_STROKE, "crash": L.GAP0}.get(variant, FULL)
    crash = variant == "crash"
    ax = L.GAP_AIM_X
    right = [] if crash else leaf_to_gap_moves()
    t_low = _duration(right)
    approach = left_approach()
    t_wait = max(0.1, t_low - _duration(approach))
    left = [_w(t_wait, *PARK_L, OPEN, "wait aside", YAW_L)] + approach
    x1 = push_x(stroke, 0.0 if variant == "short" else OVERSHOOT)
    push_d = abs(x1 - PUSH_START_X) / PUSH_SPEED
    left.append(_w(push_d, x1, 0.0, PUSH_TCP, CLOSED, "push the sensor up to the leaf", YAW_L))
    left.append(_w(0.5, x1, 0.0, PUSH_TCP, CLOSED, "settle: the leaf is pinched", YAW_L))
    t_pinch = _duration(left)
    if not crash:
        if t_pinch > _duration(right):
            right.append(_w(t_pinch - _duration(right), ax, HANDLE_Y_GAP, GRIP_TABLE, GRIP_R,
                            "hold the leaf upright until it is pinched", YAW_R))
        right.append(_w(1.5, ax, HANDLE_Y_GAP, GRIP_TABLE, OPEN_R, "let go of the leaf (pinched)", YAW_R))
        back_y, up_z = HANDLE_Y_GAP - 0.040, GRIP_TABLE + 0.050
        right.append(_w(1.0, ax, back_y, up_z, OPEN_R, "up and back, clear of the handle", YAW_R))
        right.append(_w(1.5, ax, back_y, up_z, CLOSED, "close the fingers into a block", YAW_R))
    t_right_clear = _duration(right)
    if t_right_clear > t_pinch:
        left.append(_w(t_right_clear - t_pinch, x1, 0.0, PUSH_TCP, CLOSED, "hold the push until the right is clear",
                       YAW_L))
    left += left_retreat(x1)
    wait = _duration(left) - _duration(right) - LEVER_LEAD  # the right sets off while the left hand is moving aside
    if wait > 0:
        right.append(_w(wait, ax, HANDLE_Y_GAP - 0.040, GRIP_TABLE + 0.050, CLOSED, "wait for the left hand to clear",
                        YAW_R))
    right += lever_moves()
    if variant not in ("leaf_stays", "crash"):
        right += leaf_out_moves(TABLE_DROP if variant == "leaf_table" else None)
    else:
        right.append(_w(1.5, *PARK_R[:2], PARK_R[2], OPEN, "aside", YAW_R))
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short       the sensor pushed 9.0 mm: 3.0 mm gap, lever shut (0.25)
      crash       no leaf, the sensor pushed to the dog, lever shut (0, latched)
      leaf_stays  as the demo but the leaf is left standing in the gap (0.5)
      leaf_table  the leaf let go over bare table, not in its tray (0.75)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
