"""On the station table (a kitchen counter), real size: a Joseph Joseph Flip-Up draining board (319 x 400 mm) with its
plate rack folded flat, the sink unit's steel back upstand behind it, the tiled splashback, the sink's steel rim and the
mixer tap to its right. Two washed Corelle 6.75 in side plates (171 mm across, 22 mm deep, 0.130 kg, real dished
shape) lean on the upstand to the board's left, faces toward the robot, propped there because the rack was folded.

Goal: lift the plate rack until it locks upright, then stand both plates on edge in it, each between its own pair of
prongs (different slots), resting on the board's deck and leaning on a prong, hands off.

The plates are mechkit compounds (a well, a sloping wall of 16 boxes and a flat rim of 24); the board is a fixed
mechkit articulation whose one link, ``rack`` (a hinge bar and seven prongs), turns about a hinge along x from flat (0)
to upright (90 degrees, a moulded stop at 92). The lock is a rule on the hinge's drive (``_rack_lock``): within 5
degrees of upright it holds the rack at 90 degrees; elsewhere the rack sinks back flat under its own weight. The
scorer reads the plates' poses and velocities, the rack's angle, and the arms (rigstate.py, the let-go check).

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
from scoring import BOARD, PLATES, RACK  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Put the two washed plates in the draining board's plate rack: lift the rack until it locks upright, "
               "then stand each plate on edge in a different slot, and let go.")


# -- the parts --------------------------------------------------------------------------------------------------------
def _placed_box(center_uvw, size, axes_uvw):
    """A box given in plate coordinates (centre, size along its own axes, those axes as columns in u, v, w), drawn
    in the plate's base frame (leaning as found)."""
    c = L.R_DRAWN @ (np.asarray(center_uvw, dtype=float) - np.array([0.0, 0.0, L.W_ORIGIN]))
    rot = L.R_DRAWN @ np.asarray(axes_uvw, dtype=float)
    return mk.box(tuple(float(v) for v in c), size, L.PL_WHITE, quat=tuple(float(v) for v in mk.matrix_to_quat(rot)))


def plate_shapes():
    """The well (a disc), the sloping wall (16 boxes) and the flat rim (24 boxes), drawn leaning as found."""
    shapes = []
    well_c = L.R_DRAWN @ np.array([0.0, 0.0, L.WELL_T / 2 - L.W_ORIGIN])
    shapes.append(mk.cylinder(tuple(float(v) for v in well_c), L.WELL_R, L.WELL_T, axis="Z", color=L.PL_WHITE,
                              quat=tuple(float(v) for v in mk.matrix_to_quat(L.R_DRAWN))))
    # sloping wall: mid-surface from (r 55, w 2) to (r 72.5, w 20)
    r0, w0, r1, w1 = L.WELL_R, L.WELL_T / 2, L.MARLY_R0 + 0.001, L.MARLY_W
    run, rise = r1 - r0, w1 - w0
    length = math.hypot(run, rise)
    rm, wm = (r0 + r1) / 2, (w0 + w1) / 2
    width = 2 * math.pi * rm / L.N_WALL + 0.001
    for i in range(L.N_WALL):
        a = 2 * math.pi * i / L.N_WALL
        er = np.array([math.cos(a), math.sin(a), 0.0])
        et = np.array([-math.sin(a), math.cos(a), 0.0])
        slope = (run * er + rise * np.array([0.0, 0.0, 1.0])) / length
        nrm = np.cross(slope, et)
        shapes.append(_placed_box(rm * er + np.array([0.0, 0.0, wm]), (length, width, L.PL_WALL_T),
                                  np.column_stack([slope, et, nrm])))
    # flat rim: radius 71.5 .. 85.5 mm, w 18 .. 22 mm
    rr = (L.MARLY_R0 + L.PL_R) / 2
    rim_w = 2 * L.PL_R * math.sin(math.pi / L.N_MARLY)
    for i in range(L.N_MARLY):
        a = 2 * math.pi * i / L.N_MARLY
        er = np.array([math.cos(a), math.sin(a), 0.0])
        et = np.array([-math.sin(a), math.cos(a), 0.0])
        shapes.append(_placed_box(rr * er + np.array([0.0, 0.0, L.MARLY_W]),
                                  (L.PL_R - L.MARLY_R0, rim_w, L.MARLY_W1 - L.MARLY_W0),
                                  np.column_stack([er, et, np.array([0.0, 0.0, 1.0])])))
    return shapes


BOARD_X = (L.BD_X0 + L.BD_X1) / 2  # the board's base frame: on the counter at the deck's centre


def board_shapes():
    return [mk.box((0.0, 0.0, L.DECK / 2), (L.BD_X1 - L.BD_X0, L.BD_Y1 - L.BD_Y0, L.DECK), L.BD_GREY)]


def rack_shapes():
    """The hinge bar and seven prongs, drawn folded flat (prongs pointing -y along the deck), board frame."""
    bar = mk.box((L.RACK_X - BOARD_X, L.HINGE_Y, L.HINGE_Z), (L.BAR_X1 - L.BAR_X0, L.BAR_T, L.BAR_T), L.RACK_WHITE)
    prongs = [mk.box((x - BOARD_X, L.HINGE_Y - L.BAR_T / 2 - L.PRONG_L / 2, L.HINGE_Z),
                     (L.PRONG_T, L.PRONG_L, L.PRONG_T), L.RACK_WHITE) for x in L.PRONG_X]
    return [bar, *prongs]


RACK_LINK = mk.hinge(RACK, "-X", (L.RACK_X - BOARD_X, L.HINGE_Y, L.HINGE_Z), rack_shapes(), upper=L.Q_STOP, lower=0.0,
                     mass=L.RACK_MASS, spring=L.LOCK_SPRING, damping=L.HINGE_DAMPING, color=L.RACK_WHITE)
DRAINER = mk.assembly(BOARD, origin=(BOARD_X, 0.0, L.TOP), shapes=board_shapes(), joints=[RACK_LINK],
                      color=L.BD_GREY, friction=L.BD_FRICTION, tight_fit=True)


def upstand_shapes():
    return [mk.box((0.0, 0.0, L.UP_H / 2), (L.UP_T, L.UP_Y1 - L.UP_Y0, L.UP_H), L.UP_STEEL)]


def wall_shapes():
    return [mk.box((0.0, 0.0, L.WALL_H / 2), (L.WALL_T, L.WALL_Y1 - L.WALL_Y0, L.WALL_H), L.WALL_TILE)]


def sink_shapes():
    """The sink's steel rim on the counter round a dark bowl opening (the bowl itself is not modelled)."""
    w, d = L.SINK_X1 - L.SINK_X0, L.SINK_Y1 - L.SINK_Y0
    rx, ry = L.SINK_HOLE
    h = L.SINK_H
    return [mk.box((0.0, (d - ry) / 2, h / 2), (w, ry, h), L.UP_STEEL),
            mk.box((0.0, -(d - ry) / 2, h / 2), (w, ry, h), L.UP_STEEL),
            mk.box(((w - rx) / 2, 0.0, h / 2), (rx, d - 2 * ry, h), L.UP_STEEL),
            mk.box((-(w - rx) / 2, 0.0, h / 2), (rx, d - 2 * ry, h), L.UP_STEEL),
            mk.box((0.0, 0.0, h / 4), (w - 2 * rx, d - 2 * ry, h / 2), L.SINK_DARK)]


def tap_shapes():
    """The mixer tap: a column and a spout reaching toward the robot over the bowl."""
    return [mk.cylinder((0.0, 0.0, L.TAP_H / 2), L.TAP_R, L.TAP_H, axis="Z", color=L.UP_STEEL),
            mk.box((-L.TAP_SPOUT / 2, 0.0, L.TAP_H - 0.010), (L.TAP_SPOUT, 0.022, 0.020), L.UP_STEEL)]


UPSTAND = mk.assembly("upstand", origin=(L.UP_X, L.UP_Y, L.TOP), shapes=upstand_shapes(), color=L.UP_STEEL,
                      friction=L.BD_FRICTION, tight_fit=True)
WALL = mk.assembly("wall", origin=(L.WALL_X0 + L.WALL_T / 2, (L.WALL_Y0 + L.WALL_Y1) / 2, L.TOP),
                   shapes=wall_shapes(), color=L.WALL_TILE)
SINK = mk.assembly("sink", origin=((L.SINK_X0 + L.SINK_X1) / 2, (L.SINK_Y0 + L.SINK_Y1) / 2, L.TOP),
                   shapes=sink_shapes(), color=L.UP_STEEL)
TAP = mk.assembly("tap", origin=(L.TAP_X, L.TAP_Y, L.TOP), shapes=tap_shapes(), color=L.UP_STEEL)
PLATE_PARTS = {n: mk.assembly(n, origin=(L.LEAN_X, L.PL_Y[n], L.TOP + L.LEAN_Z), mass=L.PL_MASS,
                              shapes=plate_shapes(), color=L.PL_WHITE, friction=L.PL_FRICTION, tight_fit=True)
               for n in PLATES}
OBJECTS = [WALL, UPSTAND, SINK, TAP, DRAINER, *PLATE_PARTS.values()]
SPECS = {BOARD: DRAINER, **PLATE_PARTS}


# -- the lock (a rule on the hinge's drive; simulator only) ------------------------------------------------------------
@mk.on_step
def _rack_lock(states) -> None:
    """Within 5 degrees of upright the drive target is upright (the lock holds it at 90 degrees); elsewhere it sits
    LOCK_BIAS below the rack's angle, so an unlocked rack sinks back flat under its own weight."""
    s = states.get(BOARD)
    if s is None:
        return
    q = s.joint(RACK)
    mk.set_targets(DRAINER, RACK, mk.detent_target(q, (L.Q_UP,), L.LOCK_CAPTURE, L.LOCK_BIAS))


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly as a box and drops it straight down; it knows nothing of a hinge,
# a lock, a dished plate or a rack's prongs. Offline only (never in the simulator), this model gives the scorer:
# (1) the rack: a gripper that closes below 15 % with its pads on the middle prong's tip turns the rack with it (its
# angle follows the pads about the hinge, 0 to 92 degrees); let go within 5 degrees of upright it locks at 90, else it
# falls flat. (2) a plate whose top rim lies between the open pads while they close below 15 % is gripped, stood upright
# and carried with the pads (never below the deck or the counter); opened above 25 % it stays standing if the rack is
# up, its bottom is between one slot's prongs on the deck and the prongs cross it (centre line within 70 mm of the
# prong row), and then tips onto the near prong at the computed lean (layout.lean_in_slot, about 15 degrees); else it
# falls flat onto what is under it.
GRIP_CLOSE = 0.15
GRIP_OPEN = 0.25
PRONG_GRIP_R = 0.087  # the pads pinch the middle prong this far from the hinge line (12 mm from its tip)
PRONG_TIP_OFF = 0.003  # the prong's axis sits this far above the pad tips


def rack_pinch_point(q: float) -> np.ndarray:
    """World point (counter z 0) on the middle prong's axis PRONG_GRIP_R from the hinge line, rack at ``q`` (rad)."""
    return np.array([L.RACK_X, L.HINGE_Y - PRONG_GRIP_R * math.cos(q), L.HINGE_Z + PRONG_GRIP_R * math.sin(q)])


def board_state(q: float, source: str = "offline"):
    return mk.state_from_poses(DRAINER, (BOARD_X, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0), joints={RACK: float(q)},
                               source=source)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.mode = {n: "lean" for n in PLATES}
        self.pos = {n: np.array([L.LEAN_X, L.PL_Y[n], L.LEAN_Z]) for n in PLATES}  # counter z 0
        self.rot = {n: np.eye(3) for n in PLATES}
        self.held: dict = {}  # plate -> (arm, dy, top over the pad tips)
        self.vz = {n: 0.0 for n in PLATES}
        self.foot: dict = {}  # plate -> x of its lowest point once it first rests on the deck in the hand
        self.q = 0.0
        self.rack_arm: str | None = None

    # -- the rack
    def _rack_step(self, side, tips, opening) -> bool:
        """True when this arm is busy with the rack this step."""
        if self.rack_arm == side:
            if opening >= GRIP_OPEN:
                self.rack_arm = None
                self.q = L.Q_UP if self.q >= L.Q_UP - L.LOCK_CAPTURE else 0.0
                return True
            p = np.asarray(tips, dtype=float) + np.array([0.0, 0.0, PRONG_TIP_OFF - L.TOP])
            self.q = float(np.clip(math.atan2(p[2] - L.HINGE_Z, L.HINGE_Y - p[1]), 0.0, L.Q_STOP))
            return True
        if self.rack_arm is None and opening <= GRIP_CLOSE:
            p = np.asarray(tips, dtype=float) + np.array([0.0, 0.0, PRONG_TIP_OFF - L.TOP])
            c = rack_pinch_point(self.q)
            if abs(p[0] - c[0]) <= 0.006 and math.hypot(p[1] - c[1], p[2] - c[2]) <= 0.008:
                self.rack_arm = side
                return True
        return False

    # -- the plates
    def _support(self, x: float, y: float) -> float:
        return L.DECK if (L.BD_X0 <= x <= L.BD_X1 and L.BD_Y0 <= y <= L.BD_Y1) else 0.0

    def _drop(self, name: str) -> None:
        """The plate let go of: it stays standing in a slot (and tips onto the near prong), or falls flat."""
        pts = L.plate_points(self.pos[name], self.rot[name])
        low = float(pts[:, 2].min())
        band = pts[pts[:, 2] <= low + L.BOTTOM_BAND]
        x0, x1 = float(band[:, 0].min()), float(band[:, 0].max())
        y = float(self.pos[name][1])
        if name in self.foot:  # leaned over in the hand about its foot: it is let go where its foot stands
            foot_x = self.foot.pop(name)
            x0, x1 = foot_x - (x1 - x0) / 2, foot_x + (x1 - x0) / 2
        else:
            foot_x = (x0 + x1) / 2
        slot = next((k for k in range(L.N_SLOTS) if x0 >= L.SLOT_LO[k] - L.SLOT_TOL and x1 <= L.SLOT_HI[k] +
                     L.SLOT_TOL), None)
        on_deck = self._support(foot_x, y) == L.DECK and low - L.DECK <= 0.006
        crossed = 0.0 < y - L.HINGE_Y <= 0.070
        self.mode[name] = "down"
        if self.q >= math.radians(L.RACK_UP_DEG) and slot is not None and on_deck and crossed:
            lean = L.lean_in_slot(slot, foot_x, y)
            self.rot[name] = L.upright_rot(lean)
            self.pos[name] = L.standing_origin(foot_x, y, lean)
            self.vz[name] = -math.sqrt(2 * L.G * (low - L.DECK)) if low - L.DECK > 0.003 else 0.0
            return
        support = self._support(float(self.pos[name][0]), y)
        flat = L.upright_rot(90.0)
        p = L.CLOUD0 @ flat.T
        z0 = float(self.pos[name][2])
        self.rot[name] = flat
        self.pos[name] = np.array([self.pos[name][0], y, support - float(p[:, 2].min())])
        drop = z0 - float(self.pos[name][2])
        self.vz[name] = -math.sqrt(2 * L.G * max(drop, 0.0)) if drop > 0.003 else 0.0

    def step(self, arms) -> None:
        for n in PLATES:
            self.vz[n] = 0.0
        read = rigstate.read_arms(arms)
        if read is None:
            return
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            if self._rack_step(side, tips, opening):
                continue
            held = [n for n, h in self.held.items() if h[0] == side]
            if held:
                n = held[0]
                if opening >= GRIP_OPEN:
                    del self.held[n]
                    self._drop(n)
                    continue
                _, dy, top_over_tip = self.held[n]
                top = float(tips[2]) - L.TOP + top_over_tip
                up = L.upright_rot(0.0)
                pts = L.CLOUD0 @ up.T
                o = np.array([float(tcp[0]) + 0.009, float(tcp[1]) + dy, top - float(pts[:, 2].max())])
                support = max(self._support(o[0], o[1]), 0.0)
                low = o[2] + float(pts[:, 2].min())
                if low < support:
                    o[2] += support - low
                self.pos[n], self.rot[n] = o, up
                if n not in self.foot and support == L.DECK and low <= L.DECK + 0.004:
                    band = pts[pts[:, 2] <= pts[:, 2].min() + L.BOTTOM_BAND]
                    self.foot[n] = float(o[0]) + float(band[:, 0].min() + band[:, 0].max()) / 2
                continue
            if opening > GRIP_CLOSE:
                continue
            for n in PLATES:
                if self.mode[n] != "lean" or n in self.held:
                    continue
                y = L.PL_Y[n]
                tip_z = float(tips[2]) - L.TOP
                between = abs(L.TOP_X - float(tcp[0])) <= rigstate.FULL_GAP / 2 - 0.004
                if between and abs(float(tcp[1]) - y) <= 0.030 and L.TOP_Z - 0.030 <= tip_z <= L.TOP_Z + 0.002:
                    # stood upright by the pads on its foot: its top reaches the pads, carried from then on
                    up = L.upright_rot(0.0)
                    pts = L.CLOUD0 @ up.T
                    stand_top = float(pts[:, 2].max() - pts[:, 2].min())
                    self.held[n] = (side, y - float(tcp[1]), stand_top - tip_z)
                    self.mode[n] = "held"
                    self.rot[n] = up
                    self.pos[n] = np.array([float(tcp[0]) + 0.009, y, -float(pts[:, 2].min())])
                    break

    def states(self) -> dict:
        out = {}
        for n in PLATES:
            vel = (0.0, 0.0, self.vz[n], 0.0, 0.0, 0.0)
            pose = (*(self.pos[n] + np.array([0.0, 0.0, L.TOP])), *mk.matrix_to_quat(self.rot[n]))
            out[n] = mk.state_from_poses(PLATE_PARTS[n], pose, vel, source="offline")
        out[BOARD] = board_state(self.q)
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(PLATES[0]), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, rack deg, p1 low mm, p1 tilt, p2 low mm, p2 tilt]


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: every plate's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The rack up (85 degrees or more) and both plates standing on edge in different slots (lowest point within 5 mm
    of the deck, bottom between one slot's prongs, plane within 30 degrees of vertical, centre line 30 to 70 mm beyond
    the prong row), both grippers 30 mm clear of the plates and the rack and everything still for 1 s, no latched
    fall; held 1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the
    offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in (*PLATES, BOARD)}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
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
    row["source"] = getattr(parts.get(PLATES[0]), "source", None)
    d = result.details
    if d.get("valid_state") and _TRACKER.steps % 2 == 0:
        _HIST.append([_TRACKER.steps, d["rack_deg"], d["p1_low_mm"], d["p1_tilt_deg"], d["p2_low_mm"],
                      d["p2_tilt_deg"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a plate fell (downward speed 0.9 m/s at a step) or left the counter, if the rack is not up, or unless
    both grippers are 30 mm clear of the plates and the rack and everything is still for 1 s; 1.0 both plates in
    different slots; 0.5 one slot holds a plate (or both plates share it) and every plate is upright; 0.25 no plate
    in a slot and every plate upright, or one plate in a slot and the other lying down; 0 otherwise. The score of the
    latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap; a 4 mm rim or an 8 mm prong is gripped
# with the fingers commanded closed (each pushes with 5000 N/m x 2 to 4 mm: the part ends centred between them).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
CLOSED = 0.0
PRONG_OPEN = -0.018  # a 36 mm gap round the prong: the fingers' outsides stay 8 mm off the neighbouring prongs
LET_GO_GAP = -0.020  # a 40 mm gap at a plate's let-go
YAW = -math.pi / 2  # the fingers close along x: across the plate's thickness, and across a prong

# -- the right arm: the rack
RACK_TCP_OVER = PAD_TIP - PRONG_TIP_OFF  # TCP over the prong's axis: the pad tips 3 mm below it (1 mm over the deck)
YAW_R = math.pi / 2  # the right arm's fingers also close along x (the same grip turned half a turn: in its reach)
RACK_ARC_DEG = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
RACK_PUSH_DEG = 92.0  # to the stop: inside the lock's 5 degree capture
RIGHT_PARK = (0.15, -0.25, 0.20)


def _w(d, x, y, z, finger, label, yaw=YAW):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _r(d, x, y, z, finger, label):
    return _w(d, x, y, z, finger, label, YAW_R)


def _rack_point(deg: float):
    p = rack_pinch_point(math.radians(deg))
    return float(p[0]), float(p[1]), float(p[2] + RACK_TCP_OVER)


RACK_LEAD_Y = 0.012  # open-loop +y lead at the top of the arc so the gripped prong reaches >= 88 deg on GPU


def rack_moves(top_deg: float = RACK_PUSH_DEG):
    """Over the middle prong's tip, down round it, close, swing it up about the hinge to ``top_deg``, let go, up."""
    x, y, z = _rack_point(0.0)
    plan = [
        _r(2.0, x, y, z + 0.060, PRONG_OPEN, "over the folded rack's middle prong"),
        _r(1.5, x, y, z, PRONG_OPEN, "down round the prong's tip"),
        _r(0.5, x, y, z, PRONG_OPEN, "settle"),
        _r(1.5, x, y, z, CLOSED, "close on the prong"),
        _r(0.5, x, y, z, CLOSED, "settle the grip"),
    ]
    for deg in [d for d in RACK_ARC_DEG if d < top_deg] + [top_deg]:
        plan.append(_r(0.8, *_rack_point(deg), CLOSED, f"swing the rack up to {deg:.0f} degrees"))
    tx, ty, tz = _rack_point(top_deg)
    ty_push = ty + (RACK_LEAD_Y if top_deg >= 85.0 else 0.0)
    plan += [
        _r(0.8, tx, ty_push, tz, CLOSED, "push onto the stop: overcome arm trail"),
        _r(0.6, tx, ty_push, tz, CLOSED, "hold it on the stop: the lock takes it"),
        _r(1.5, tx, ty_push, tz, PRONG_OPEN, "let go"),
        _r(1.0, tx, ty - 0.050, tz, PRONG_OPEN, "back along y: the open pads slide off the prong's sides"),
        _r(1.0, tx, ty - 0.050, 0.150, PRONG_OPEN, "up"),
        _r(2.0, *RIGHT_PARK, OPEN, "aside"),
    ]
    return plan


# -- the left arm: the plates
GRIP_DEPTH = 0.012  # the pad tips end 12 mm below a plate's top rim (on the flat rim, 71.5 to 85.5 mm radius)
STAND_H = L.PL_D  # an upright plate's height
GRIP_Z = STAND_H - GRIP_DEPTH + PAD_TIP  # 0.172: the TCP over the counter when the plate stands on it
HOVER_Z = 0.200  # open pads 19 mm over the leaning plate's top (0.168)
CARRY_CLEAR = 0.014  # the hanging plate's lowest point this far over the deck while it is carried
CARRY_Z = round(L.DECK + CARRY_CLEAR + STAND_H - GRIP_DEPTH + PAD_TIP, 4)  # 0.198
SET_GAP = 0.002  # commanded height of the plate's lowest point over the deck at the let-go (the hand sags under load)
SET_Z = round(L.DECK + SET_GAP + STAND_H - GRIP_DEPTH + PAD_TIP, 4)  # 0.186
ENTRY_Z = round((CARRY_Z + SET_Z) / 2, 4)
X_PICK = L.FOOT_X - 0.002  # over the plate's foot: the closing pads stand it up there (its rim's mid-plane)
RIM_TO_FOOT = 0.0  # an upright plate's bottom is centred under its rim's mid-plane (the TCP's x)
PLACE_BIAS = 0.0  # a GPU correction to where the bottom is set (on top of layout.FOOT_BIAS); tune on the GPU
LEAN_SHIFT = 0.020  # still gripped on the deck, the hand draws back toward the robot this far: the plate tips about
# its foot about 7 degrees, and let go it tips on to rest on the near prong (about 15 degrees, layout.SLOT_LEAN_DEG)
SLOT_FOR = {"plate_1": 2, "plate_2": 5}
WAIT_RACK = 6.0  # plate 1 waits, lifted off the upstand, until the rack is locked and the right hand has left
LEFT_PARK = (0.20, 0.34, 0.205)  # over the counter left of the board, clear of the plates and the rack


def plate_moves(name: str, slot: int, wait: float = 0.0, place_x: float | None = None, lean: bool = True):
    """The left arm: over the plate's rim, down round it, close (the pads stand it upright), lift it, carry it along x
    beside the rack (clear of the prongs), slide it along y into its slot, lower it to the deck, lean it toward the
    robot, let go, up."""
    px, py = X_PICK, L.PL_Y[name]
    sx = (L.SLOT_X[slot] + L.FOOT_BIAS + PLACE_BIAS) if place_x is None else place_x
    lx = sx - LEAN_SHIFT if lean else sx
    plan = [
        _w(2.0, px, py, HOVER_Z, OPEN, f"over {name}'s top rim"),
        _w(1.0, px, py, GRIP_Z + 0.020, OPEN, "down to just above the rim"),
        _w(1.5, px, py, GRIP_Z, OPEN, "slowly down round the rim"),
        _w(0.5, px, py, GRIP_Z, OPEN, "settle"),
        _w(1.5, px, py, GRIP_Z, CLOSED, "close on the rim: it stands upright between the pads"),
        _w(0.5, px, py, GRIP_Z, CLOSED, "settle the grip"),
        _w(2.0, px, py, CARRY_Z, CLOSED, "lift it straight up off the upstand"),
    ]
    if wait > 0:
        plan.append(_w(wait, px, py, CARRY_Z, CLOSED, "wait, held up, for the rack to lock and the right hand to go"))
    plan += [
        _w(2.0, sx, py, CARRY_Z, CLOSED, f"carry it along x to slot {slot}'s line, beside the rack"),
        _w(2.5, sx, L.PLACE_Y, CARRY_Z, CLOSED, f"slide it along y into slot {slot}, between the prongs"),
        _w(1.5, sx, L.PLACE_Y, ENTRY_Z, CLOSED, "lower it"),
        _w(1.5, sx, L.PLACE_Y, SET_Z, CLOSED, "lower it slowly to the deck"),
        _w(0.5, sx, L.PLACE_Y, SET_Z, CLOSED, "settle"),
    ]
    if lean:
        plan.append(_w(1.5, lx, L.PLACE_Y, SET_Z, CLOSED, "draw the hand back: the plate tips about its foot"))
    plan += [
        _w(1.5, lx, L.PLACE_Y, SET_Z, LET_GO_GAP, "let go: it leans on the near prong"),
        _w(1.5, lx, L.PLACE_Y, HOVER_Z + 0.005, LET_GO_GAP, "straight up"),
    ]
    return plan


# Wrong ways (counterfactuals, offline only).
VARIANTS = ("one_plate", "rack_only", "unlocked")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the right arm lifts the rack and locks it while the left arm picks up plate_1; plate_1 goes to slot 1
    once the rack is locked, then plate_2 to slot 4; both hands park aside."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = rack_moves(75.0 if variant == "unlocked" else RACK_PUSH_DEG)
    if variant == "rack_only":
        return {"right": right}
    left = plate_moves("plate_1", SLOT_FOR["plate_1"], wait=WAIT_RACK)
    if variant != "one_plate":
        left += plate_moves("plate_2", SLOT_FOR["plate_2"])
    left.append(_w(2.5, *LEFT_PARK, OPEN, "aside, clear of both plates and the rack"))
    return {"left": left, "right": right}


def demo():
    """Right arm: the rack. Left arm: the plates. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling
    check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      one_plate  only plate_1 is moved: plate_2 stays leaning on the upstand (0.5)
      rack_only  the rack is lifted and locked, no plate is moved (0.25)
      unlocked   the rack is let go at 75 degrees: it falls back flat, and the plates set on it fall over (0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
