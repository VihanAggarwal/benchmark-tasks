"""On the station table, real size: a fixed child's snowboard top (1100 x 240 x 10 mm) lying across the table along y,
nose to the left, heel edge 18 cm in front of the arm bases: two white strips with a 4 mm black centre line between
them at x 30 cm, a small red reference cross at its middle (y 0) and 16 M6 inserts (8 mm bores, 9 mm deep) on the
4 x 4 pattern, two rows across the board at x 28 and 32 cm and four rows along it on each half at |y| 14, 18, 22 and
26 cm. Two loose child's bindings (0.35 kg each: baseplate, sidewalls, highback and a 4 x 4 disc with four slots)
lie parked flat on the board's two ends, toes toward the ends; each disc is preset in its baseplate, +15 deg on the
nose binding and -15 deg on the tail binding. Beyond the board's toe edge, two fixed holders each hold four M6
binding screws with washers, head up.

Goal: mount the bindings for a 40 cm duck stance centred on the cross, heels toward the robot, and screw them down.
Each binding's slots line up over four inserts only with its disc square to the board (within about 9 deg: the slots
are long; the scorer asks for 3); then its baseplate is at its preset angle. The 40 cm stance puts both discs on the
middle squares (inserts at |y| 18 and 22 cm). A screw dropped through a slot into an insert and turned one full turn
clockwise while seated is driven (welded where it stands); a binding whose four screws are driven is clamped (welded
flat). See scoring.py for the rules and the tiers.

The bindings, screws, board and holders are mechkit assemblies (mechkit.py next to this file); the thread and clamp
rules run in a mechkit on_step callback in the simulator, and the arms are read through rigstate.py. Neither changes
what the embodiment does.

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

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built as compound bodies)
import rigstate  # noqa: E402
from scoring import BINDINGS, SCREWS, Fastening, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# 1.0 must hold for 1 s on top of the scorer's own 1 s of release and stillness, so the trial cannot end while a
# gripper holds or touches a part.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Mount this rental snowboard's bindings for a 40 cm duck stance centred on the red cross, heels on the "
               "edge nearest you, and screw both down.")


def _turned(shapes, angle: float):
    """Shapes given in a baseplate frame, turned ``angle`` (rad) about the binding frame's z axis."""
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for sh in shapes:
        x, y, z = sh.center
        out.append(mk.box((c * x - s * y, s * x + c * y, z), sh.size, sh.color, ypr=(angle, 0.0, 0.0)))
    return out


# -- the board: three strips over a dark 1 mm floor layer, the insert bores cut through the strips ------------------
def _strip_holes(strip_cx: float, insert_x: float):
    return [(insert_x - strip_cx, y, L.INSERT_HOLE, L.INSERT_HOLE)
            for y in (*L.INSERT_YS_NOSE, *L.INSERT_YS_TAIL)]


_HEEL_CX = L.BOARD_X0 + L.STRIP_W / 2 - L.BOARD_CX  # strip centres in the board frame
_TOE_CX = L.BOARD_X1 - L.STRIP_W / 2 - L.BOARD_CX
_Z_STRIP = L.FLOOR_LAYER + L.INSERT_DEPTH / 2
BOARD = mk.assembly("board", origin=(L.BOARD_CX, 0.0, L.TOP), color=L.WHITE, tight_fit=True, shapes=[
    mk.box((0.0, 0.0, L.FLOOR_LAYER / 2), (L.BOARD_W, L.BOARD_LEN, L.FLOOR_LAYER), L.INSERT_GREY),
    *mk.plate_with_holes((_HEEL_CX, 0.0, _Z_STRIP), (L.STRIP_W, L.BOARD_LEN),
                         _strip_holes(_HEEL_CX, L.INSERT_XS[0] - L.BOARD_CX), L.INSERT_DEPTH, L.WHITE),
    mk.box((L.LINE_X - L.BOARD_CX, 0.0, _Z_STRIP), (L.LINE_W, L.BOARD_LEN, L.INSERT_DEPTH), L.BLACK),
    *mk.plate_with_holes((_TOE_CX, 0.0, _Z_STRIP), (L.STRIP_W, L.BOARD_LEN),
                         _strip_holes(_TOE_CX, L.INSERT_XS[1] - L.BOARD_CX), L.INSERT_DEPTH, L.WHITE),
])
CROSS = [  # the strip along y sits on the one along x, so it shows over the black centre line
    Box("cross_x", pos=(L.CROSS_X, L.CROSS_Y, L.BOARD_TOP + L.CROSS_T / 2), size=(L.CROSS_LEN, L.CROSS_W, L.CROSS_T),
        color=L.RED),
    Box("cross_y", pos=(L.CROSS_X, L.CROSS_Y, L.BOARD_TOP + 1.5 * L.CROSS_T), size=(L.CROSS_W, L.CROSS_LEN, L.CROSS_T),
        color=L.RED),
]

# -- the bindings (loose; the disc frame is the binding's frame) ---------------------------------------------------
_DISC = [
    *mk.plate_with_holes((0.0, 0.0, L.FLOOR_T / 2), (L.DISC_CORE, L.DISC_CORE),
                         [(sx, sy, L.SLOT_LEN, L.SLOT_W) for sx, sy in L.SLOT_CENTRES], L.FLOOR_T, L.DISC_GREY),
    mk.box((L.DISC_TAB_AT, 0.0, L.FLOOR_T / 2), (L.DISC_TAB[0], L.DISC_TAB[1], L.FLOOR_T), L.DISC_GREY),
    mk.box((-L.DISC_TAB_AT, 0.0, L.FLOOR_T / 2), (L.DISC_TAB[0], L.DISC_TAB[1], L.FLOOR_T), L.DISC_GREY),
    mk.box((0.0, L.DISC_TAB_AT, L.FLOOR_T / 2), (L.DISC_TAB[1], L.DISC_TAB[0], L.FLOOR_T), L.DISC_GREY),
    mk.box((0.0, -L.DISC_TAB_AT, L.FLOOR_T / 2), (L.DISC_TAB[1], L.DISC_TAB[0], L.FLOOR_T), L.DISC_GREY),
]
_SW_LEN = L.SIDEWALL_U[1] - L.SIDEWALL_U[0]
_SW_U = (L.SIDEWALL_U[0] + L.SIDEWALL_U[1]) / 2
_HB_U = (L.HIGHBACK_U[0] + L.HIGHBACK_U[1]) / 2  # -0.110
_BASEPLATE = [  # baseplate frame: u heel to toe, v across, z up from the underside
    *mk.plate_with_hole((0.0, 0.0, L.FLOOR_T / 2), (L.BASE_LEN, L.BASE_W), (L.DISC_OPENING, L.DISC_OPENING),
                        L.FLOOR_T, color=L.BLACK),
    mk.box((_SW_U, L.SIDEWALL_V, L.FLOOR_T + L.SIDEWALL_H / 2), (_SW_LEN, L.SIDEWALL_T, L.SIDEWALL_H), L.BLACK),
    mk.box((_SW_U, -L.SIDEWALL_V, L.FLOOR_T + L.SIDEWALL_H / 2), (_SW_LEN, L.SIDEWALL_T, L.SIDEWALL_H), L.BLACK),
    mk.box((_HB_U, 0.0, L.FLOOR_T + L.HIGHBACK_H / 2), (L.HIGHBACK_U[1] - L.HIGHBACK_U[0], L.HIGHBACK_W,
                                                       L.HIGHBACK_H), L.BLACK),
]
# The highback's column, baseplate frame: from the underside to the highback's top (what the offline checker sees)
_HB_COLUMN = (np.array([_HB_U, 0.0, L.HIGHBACK_TOP / 2]),
              (L.HIGHBACK_U[1] - L.HIGHBACK_U[0], L.HIGHBACK_W, L.HIGHBACK_TOP))


class Binding(mk.Assembly):
    """A binding exactly as mechkit builds and reads it in the simulator. Only tools/offline/check_task.py calls
    the two methods below: offline a binding is its highback's column (the highback from the underside to its top),
    the part a gripper closes on to carry it, so the checker grasps that column, carries it turned with the gripper
    and drops it onto the board; ``offline_state`` rebuilds the disc frame from it."""

    def _column_local(self) -> np.ndarray:
        p = L.preset_rad(self.name)
        c, s = math.cos(p), math.sin(p)
        u, v, z = _HB_COLUMN[0]
        return np.array([c * u - s * v, s * u + c * v, z])

    def offline_box(self):
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ self._column_local()
        return tuple(float(v) for v in centre), tuple(float(v) for v in _HB_COLUMN[1]), \
            float(self.yaw + L.preset_rad(self.name))

    def offline_state(self, state):
        r = mk.quat_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
        yaw = math.atan2(r[1, 0], r[0, 0]) - L.preset_rad(self.name)
        quat = mk.yaw_quat(yaw)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ self._column_local()
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


def _binding(name: str, start) -> Binding:
    x, y, angle = start
    b = Binding(name=name, pos=(x, y, L.BOARD_TOP), color=L.BLACK, mass=L.BINDING_MASS,
                yaw=math.radians(angle) - L.preset_rad(name),
                shapes=(*_DISC, *_turned(_BASEPLATE, L.preset_rad(name))), **mk.TIGHT_FIT)
    b._check()
    return b


BINDING_NOSE = _binding("binding_nose", L.NOSE_START)
BINDING_TAIL = _binding("binding_tail", L.TAIL_START)

# -- the screws and their holders ----------------------------------------------------------------------------------
_SCREW_SHAPES = (
    mk.cylinder((0.0, 0.0, L.WASHER_T / 2), L.WASHER_R, L.WASHER_T),
    mk.cylinder((0.0, 0.0, L.WASHER_T + L.HEAD_H / 2), L.HEAD_R, L.HEAD_H),
    mk.cylinder((0.0, 0.0, -L.STUB_L / 2), L.STUB_R, L.STUB_L),
)
_HEAD_TOP = L.WASHER_T + L.HEAD_H  # 6.5 mm above the washer's underside
# Offline stand-in: the head's footprint from 8 mm below the washer (the disc's thickness: where the board's top is
# when the screw is seated) to the head's top, so a column the checker drops onto the board stands where a seated
# screw does. In a holder this column sinks 8 mm into the holder's box (an offline overlap warning only).
_SCREW_COLUMN = (np.array([0.0, 0.0, (_HEAD_TOP - L.FLOOR_T) / 2]), (2 * L.HEAD_R, 2 * L.HEAD_R, _HEAD_TOP + L.FLOOR_T))


class Screw(mk.Assembly):
    """An M6 binding screw with its washer, as mechkit builds it. Offline it is the column above (see
    ``_SCREW_COLUMN``); ``offline_state`` puts the screw's frame back on it."""

    def offline_box(self):
        origin, quat = self.spawn_pose()
        return tuple(float(v) for v in origin + _SCREW_COLUMN[0]), tuple(float(v) for v in _SCREW_COLUMN[1]), \
            float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - _SCREW_COLUMN[0]
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


def _holder(name: str, sign: float):
    ys = [sign * y for y in L.HOLDER_YS]
    cy = (min(ys) + max(ys)) / 2
    depth = L.HOLDER_TOP - L.HOLDER_BASE_T
    return mk.assembly(name, origin=(L.HOLDER_X, cy, L.TOP), color=L.HOLDER_BLUE, tight_fit=True, shapes=[
        mk.box((0.0, 0.0, L.HOLDER_BASE_T / 2), (L.HOLDER_SIZE[0], L.HOLDER_SIZE[1], L.HOLDER_BASE_T)),
        *mk.plate_with_holes((0.0, 0.0, L.HOLDER_BASE_T + depth / 2), L.HOLDER_SIZE[:2],
                             [(0.0, y - cy, L.HOLDER_SOCKET, L.HOLDER_SOCKET) for y in ys], depth),
    ])


HOLDER_NOSE = _holder("holder_nose", 1.0)
HOLDER_TAIL = _holder("holder_tail", -1.0)
SOCKETS = {f"screw_{h}{k + 1}": (L.HOLDER_X, sign * y) for h, sign in (("n", 1.0), ("t", -1.0))
           for k, y in enumerate(L.HOLDER_YS)}


def _screw(name: str) -> Screw:
    x, y = SOCKETS[name]
    s = Screw(name=name, pos=(x, y, L.TOP + L.HOLDER_TOP), color=L.STEEL, mass=L.SCREW_MASS, yaw=0.0,
              shapes=_SCREW_SHAPES, **mk.TIGHT_FIT)
    s._check()
    return s


SCREW_SPECS = [_screw(n) for n in SCREWS]
OBJECTS = [BOARD, *CROSS, BINDING_NOSE, BINDING_TAIL, HOLDER_NOSE, HOLDER_TAIL, *SCREW_SPECS]

# -- the rules: thread engagement, turn counting, drive and clamp welds ----------------------------------------------
FASTENING = Fastening()
SCREW_WELDS = {s.name: mk.weld(s, None, name=f"{s.name}_driven") for s in SCREW_SPECS}
CLAMP_WELDS = {b.name: mk.weld(b, None, name=f"{b.name}_clamp") for b in (BINDING_NOSE, BINDING_TAIL)}


def _apply(events) -> None:
    for kind, name, pose in events:
        weld = SCREW_WELDS[name] if kind == "drive" else CLAMP_WELDS[name]
        weld.fire(rel_pose=pose)
        print(f"[snowboard] {kind} {name} at episode {mk.episode()} step {mk.step()}: {json.dumps(pose)}",
              file=sys.stderr, flush=True)


@mk.on_step
def _rules(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    _apply(FASTENING.update(states, key=("sim", mk.episode(), mk.step())))


@mk.on_reset
def _rules_reset() -> None:
    FASTENING.reset()


_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


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
    """Both bindings fastened on the middle squares, every part still and both grippers clear for 1 s; held 1 s.
    This runs every control step, so it also feeds the tracker (and, offline, the fastening rules)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            FASTENING.reset()
        _EPISODE["mechkit"] = mk.episode()
    if mk.env() is None:  # offline (no simulator, so no on_step): run the rules here, once per scored step
        _apply(FASTENING.update(objects, key=("offline", mk.episode(), _TRACKER.steps + 1)))
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms, FASTENING)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details, "screws": FASTENING.status()}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 both bindings fastened on the middle squares; 0.75 both fastened elsewhere; 0.5 both aligned with four
    screws seated in each; 0.25 one; 0 otherwise (and whenever a gripper is within 30 mm of a part, anything has moved
    in the last 1 s, or a part fell off the table). The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Both arms at once, mirrored: the left arm mounts the nose binding with the nose-side screws, the right arm the tail
# binding. Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part
# (about 20 N a finger). Waypoint yaw 0 closes the fingers along world y; slerp takes the short way, so every turn
# over 90 deg is split.
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
OPEN_HB = -0.025  # 50 mm apart: 20 mm from each face of the 10 mm highback
GRIP_HB = -(0.010 - 0.008) / 2  # -0.001: a 2 mm command on the 10 mm highback
HB_GRIP_DOWN = 0.020  # the pad tips this far below the highback's top
TCP_HB = L.BOARD_T + L.HIGHBACK_TOP - HB_GRIP_DOWN + PAD_TIPS  # 0.111 above the table, binding on the board
LIFT = 0.060  # the carried binding's underside 60 mm above the board: its dipped toe stays 25 mm over the holders
HB_HOVER = 0.175
PLACE_GAP = -0.003  # commanded 3 mm into the board: the hung binding sits 6 to 10 mm low, so it rests on it
OPEN_SCREW = -0.010  # 20 mm apart: 4 mm from each side of the 12 mm head
GRIP_SCREW = -(2 * L.HEAD_R - 0.008) / 2  # -0.002: a 4 mm command on the 12 mm head
TIPS_OVER_WASHER = 0.0015  # picking and carrying a head: the pad tips 1.5 mm above the washer's top
TCP_PICK = L.HOLDER_TOP + L.WASHER_T + TIPS_OVER_WASHER + PAD_TIPS  # 0.040 above the table
DISC_TOP = L.BOARD_T + L.FLOOR_T  # 0.018 above the table, binding on the board
RELEASE_GAP = 0.0005  # let a screw go with its washer 0.5 mm above the disc
TCP_RELEASE = DISC_TOP + RELEASE_GAP + L.WASHER_T + TIPS_OVER_WASHER + PAD_TIPS  # 0.0345
# Turning a seated screw. Both GPU runs of this revision lost turn the same way: the pads, holding only the bottom
# 3.5 mm of the 5 mm head (less with the TCP riding 0.3 to 1.8 mm above its command), popped up off the head; the
# closed fingers then read 4 mm and the TCP 3.3 mm high, resting on the head's top. Once (run 2, screw_t4) the pads
# took the 6 g screw up with them, its washer rose past 3 mm and its count went back to 0. So the seated head is gripped
# with the pad tips 0.5 mm above the washer (4.5 mm of the head held; a pad that touches the washer only presses the
# seated screw down onto the disc, it cannot lift it), and the open fingers rise 2 mm before they turn back (tips
# 2.5 mm above the washer, clear of it even with the TCP 0.5 mm low, still round the head) so they do not drag it.
TIPS_GRIP = 0.0005  # gripping a seated head: the pad tips 0.5 mm above its washer's top
TCP_SEATED = DISC_TOP + L.WASHER_T + TIPS_GRIP + PAD_TIPS  # 0.033
TURN_BACK_RISE = 0.002  # open, turning back: the pad tips 2.5 mm above the washer
TCP_TURN_BACK = TCP_SEATED + TURN_BACK_RISE  # 0.035
TCP_ENTRY = TCP_RELEASE + L.STUB_L + 0.003  # stub tip 3 mm above the disc's top
TCP_CARRY = 0.100  # a carried screw's tip 70 mm above the table: over the sidewalls (58 mm), toe side only
CLEAR_H = 0.170
STROKE_DEG = 90.0  # a half stroke (one waypoint). A stroke is two of them: 180 deg
STROKES = 6  # six strokes of 180 deg: 1,080 deg commanded for the 360 the rule needs. A good stroke turns a head about
# 176 deg, so two fall just short and every screw needs a third; six leave room for one stroke lost to a pop-off and
# for one count restart (run 1 commanded 450 and screw_t3 reached 278; run 2 commanded 720 and screw_t4, restarted
# after its second stroke, reached 353). Whole multiples of 90 deg also leave every screw's (square) offline stand-in
# square to the next grip.
YAW_CENTRE = {"left": -20.0, "right": 20.0}  # deg: the wrist's joint6 is near 0 here over these inserts
YAW_PICK = 90.0  # fingers closing along x over the holder (its sockets are 30 mm apart along y)
# The binding: per arm, the TCP at the highback's centre and the gripper yaw turning with it (0 at the pick).
# Per-target trims (demo only, never the scoring): PLACE_TRIM per arm (dx m, dy m, dyaw deg) at the binding's place,
# SCREW_TRIM per insert (dx m, dy m). Hung by its highback, the binding loads the arm with 3.4 N and 0.25 N m: a static
# model of the arm (yam_arena's joint stiffness, 80 N m/rad shoulder and elbow, 10 wrist; it predicted the 2026-11-06
# GPU placement error of this task's earlier version to within 0.6 mm) puts the hanging disc 12.5 mm toward the robot
# and 2.2 mm outward of the commanded point, toe 2.5 deg down. Where it ends after the toe touches down lies between
# that and the commanded point, so the trim is half of it: +6.2 mm in x (the slots' long axis, 9.5 mm of play) and
# 1.1 mm inward (2.5 mm of play). After a GPU run: read origin in TASK_TRACE at the end and add the disc's offset
# from (0.300, +-0.200) with the sign flipped; dyaw = minus angle_err_deg.
PLACE_TRIM = {"left": (0.0062, -0.0011, 0.0), "right": (0.0062, 0.0011, 0.0)}
SCREW_TRIM: dict[str, tuple[float, float]] = {}
SIDES = {"left": ("binding_nose", L.NOSE_START, L.FRONT_TARGET_DEG, 1.0, "n"),
         "right": ("binding_tail", L.TAIL_START, L.BACK_TARGET_DEG, -1.0, "t")}
SCREW_ORDER = {"left": [("screw_n1", (0.320, 0.220)), ("screw_n2", (0.320, 0.180)), ("screw_n3", (0.280, 0.220)),
                        ("screw_n4", (0.280, 0.180))]}
SCREW_ORDER["right"] = [(n.replace("_n", "_t"), (x, -y)) for n, (x, y) in SCREW_ORDER["left"]]


def _hb_point(disc_xy, angle_deg: float) -> tuple[float, float]:
    """Where the highback's centre is for a disc centre and a binding angle."""
    a = math.radians(angle_deg)
    return disc_xy[0] + math.cos(a) * _HB_U, disc_xy[1] + math.sin(a) * _HB_U


def place_point(side: str) -> tuple[float, float, float]:
    """(TCP x, TCP y, gripper yaw in rad) at the binding's place: over the highback with the disc at its target,
    the gripper turned with the binding from yaw 0 at the pick."""
    _, (_, _, start), target, sign, _ = SIDES[side]
    dx, dy, dyaw = PLACE_TRIM[side]
    x, y = _hb_point((L.LINE_X, sign * L.MIDDLE_SQUARE_Y), target)
    return x + dx, y + dy, math.radians(target - start + dyaw)


def _at(x, y, h):
    return (x, y, L.TOP + h)


def _binding_plan(side: str) -> list[Waypoint]:
    """Carry the binding by its highback to the middle square, turning it 75 deg, and let go (20.8 s)."""
    name, (sx, sy, start), _, _, _ = SIDES[side]
    px, py = _hb_point((sx, sy), start)
    qx, qy, yaw = place_point(side)
    place_h = TCP_HB + PLACE_GAP
    return [
        Waypoint(2.5, _at(px, py, HB_HOVER), OPEN_HB, f"over the {name} highback", 0.0),
        Waypoint(1.0, _at(px, py, TCP_HB + 0.030), OPEN_HB, "down either side of the highback", 0.0),
        Waypoint(1.5, _at(px, py, TCP_HB), OPEN_HB, "slowly down the last 3 cm", 0.0),
        Waypoint(0.5, _at(px, py, TCP_HB), OPEN_HB, "settle", 0.0),
        Waypoint(1.5, _at(px, py, TCP_HB), GRIP_HB, "grip the highback", 0.0),
        Waypoint(0.5, _at(px, py, TCP_HB), GRIP_HB, "settle the grip", 0.0),
        Waypoint(2.0, _at(px, py, TCP_HB + LIFT), GRIP_HB, "lift", 0.0),
        Waypoint(4.0, _at(qx, qy, TCP_HB + LIFT), GRIP_HB, "carry it over the middle square, turning it", yaw),
        Waypoint(1.0, _at(qx, qy, place_h + 0.035), GRIP_HB, "down to 3 cm above", yaw),
        Waypoint(2.0, _at(qx, qy, place_h), GRIP_HB, "slowly down onto the board", yaw),
        Waypoint(0.8, _at(qx, qy, place_h), GRIP_HB, "settle", yaw),
        Waypoint(1.5, _at(qx, qy, place_h), OPEN_HB, "let go", yaw),
        Waypoint(0.5, _at(qx, qy, place_h), OPEN_HB, "settle", yaw),
        Waypoint(1.5, _at(qx, qy, HB_HOVER), OPEN_HB, "straight up, clear of the highback", yaw),
    ]


_ORDINAL = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


def _strokes(seat, back, hi, mid, lo) -> list[Waypoint]:
    """STROKES clockwise strokes of 180 deg on a gripped seated head, each from yaw hi to lo: turn (2 s), let go
    (0.6 s), rise 2 mm open (0.3 s), turn back open (1.2 s), down round the head (0.3 s), grip (0.8 s), settle the grip
    (0.3 s). After the last stroke: let go, rise 2 mm, turn back 90 deg open to mid (the yaw the wrist left at before),
    and the caller rises from there."""
    out = []
    for k in range(STROKES):
        done = 180 * k
        last = k == STROKES - 1
        out += [
            Waypoint(1.0, seat, GRIP_SCREW, f"turn it clockwise ({done + 90} deg)", mid),
            Waypoint(1.0, seat, GRIP_SCREW, f"turn it clockwise ({done + 180} deg"
                     + (": the rule's full turn)" if done + 180 == 360 else ")"), lo),
            Waypoint(0.6, seat, OPEN_SCREW, "let go", lo),
            Waypoint(0.3, back, OPEN_SCREW, "up 2 mm, open, clear of the washer", lo),
            Waypoint(0.6, back, OPEN_SCREW, "turn back, open"
                     + (" (so the wrist leaves at the same yaw as before)" if last else ""), mid),
        ]
        if last:
            break
        out += [
            Waypoint(0.6, back, OPEN_SCREW, "turn back, open", hi),
            Waypoint(0.3, seat, OPEN_SCREW, "down 2 mm round the head", hi),
            Waypoint(0.8, seat, GRIP_SCREW, f"grip it a {_ORDINAL[k + 1]} time", hi),
            Waypoint(0.3, seat, GRIP_SCREW, "settle the grip", hi),
        ]
    return out


def _screw_plan(side: str, screw: str, insert_xy) -> list[Waypoint]:
    """Fetch one screw, drop it through its slot into the insert, turn it 1,080 deg clockwise in six strokes (46.0 s)."""
    hx, hy = SOCKETS[screw]
    dx, dy = SCREW_TRIM.get(screw, (0.0, 0.0))
    ix, iy = insert_xy[0] + dx, insert_xy[1] + dy
    c = YAW_CENTRE[side]
    hi, mid, lo = (math.radians(c + STROKE_DEG), math.radians(c), math.radians(c - STROKE_DEG))
    pick = math.radians(YAW_PICK)
    rel = _at(ix, iy, TCP_RELEASE)
    seat = _at(ix, iy, TCP_SEATED)
    back = _at(ix, iy, TCP_TURN_BACK)
    return [
        Waypoint(2.5, _at(hx, hy, TCP_CARRY), OPEN_SCREW, f"over {screw} in its holder", pick),
        Waypoint(1.0, _at(hx, hy, TCP_PICK + 0.010), OPEN_SCREW, "down round its head", pick),
        Waypoint(0.8, _at(hx, hy, TCP_PICK), OPEN_SCREW, "slowly down the last centimetre", pick),
        Waypoint(0.3, _at(hx, hy, TCP_PICK), OPEN_SCREW, "settle", pick),
        Waypoint(1.0, _at(hx, hy, TCP_PICK), GRIP_SCREW, "grip the head", pick),
        Waypoint(0.3, _at(hx, hy, TCP_PICK), GRIP_SCREW, "settle the grip", pick),
        Waypoint(1.2, _at(hx, hy, TCP_CARRY), GRIP_SCREW, "lift it out of its socket", pick),
        Waypoint(2.0, _at(ix, iy, TCP_CARRY), GRIP_SCREW, "carry it over its insert, from the toe side", hi),
        Waypoint(1.0, _at(ix, iy, TCP_ENTRY), GRIP_SCREW, "down to 3 mm above the slot", hi),
        Waypoint(1.5, rel, GRIP_SCREW, "slowly through the slot into the insert", hi),
        Waypoint(0.3, rel, GRIP_SCREW, "settle", hi),
        Waypoint(0.6, rel, OPEN_SCREW, "let it drop onto the disc", hi),
        Waypoint(0.5, seat, OPEN_SCREW, "down 1.5 mm with it, open", hi),
        Waypoint(0.8, seat, GRIP_SCREW, "grip the seated head", hi),
        Waypoint(0.4, seat, GRIP_SCREW, "settle the grip before the first stroke", hi),
        *_strokes(seat, back, hi, mid, lo),
        Waypoint(0.8, _at(ix, iy, TCP_CARRY), OPEN_SCREW, "straight up", mid),
    ]


def _turn_first(plan: list[Waypoint], nxt: Waypoint) -> list[Waypoint]:
    """A 1 s wrist turn on the spot, halfway to the next waypoint's yaw, when the change is over 90 deg (the
    interpolation takes the short way round, and joint6 has a 240 deg window)."""
    last = plan[-1]
    turn = math.remainder(nxt.yaw - last.yaw, 2 * math.pi)
    if abs(turn) <= math.radians(STROKE_DEG) + 1e-9:
        return []
    return [Waypoint(1.0, last.pos, last.finger, "turn the wrist on the way", last.yaw + turn / 2)]


def _arm(side: str) -> list[Waypoint]:
    """One arm's plan (about 210 s); both arms run the same timeline, mirrored."""
    plan = _binding_plan(side)
    for screw, insert in SCREW_ORDER[side]:
        nxt = _screw_plan(side, screw, insert)
        plan += _turn_first(plan, nxt[0]) + nxt
    last = plan[-1]
    plan.append(Waypoint(1.5, (last.pos[0], last.pos[1], L.TOP + CLEAR_H), OPEN_SCREW, "up and clear", last.yaw))
    return plan


def demo():
    """Both arms at once. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
