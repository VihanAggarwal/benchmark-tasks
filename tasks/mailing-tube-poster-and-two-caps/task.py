"""On the station table, real size: a kraft mailing tube (51 mm bore, 54 mm outside, 457 mm = 18 in long, 0.12 kg, open at
both ends) lies along y on a 5 mm packing board at x 300 mm. A rolled poster (38 mm across, 380 mm long, 0.06 kg) lies
on the table in line with the tube's left end, its near end 16.5 mm outside the mouth. In front of each arm a black
plastic end cap (60 mm flange, 5 mm thick, a 49 mm plug 18 mm deep, 12 g) lies on its side in a low cap rest, plug
toward the tube.

Goal: the roll inside the tube (both ends 15 mm or more in) and a cap pressed onto each end of the tube until its
flange sits on the end, then hands off. The left hand picks the roll up by its middle and slides it in end first,
then takes it again by its far end and pushes it on; each
hand then lifts its cap, carries it in line with its end of the tube and presses it on, the left cap also pushing the
roll's last 40 mm home.

The tube (a ring of 16 flat staves), the roll, the caps, the board and the cap rests are mechkit compounds (mechkit.py
next to this file). There is no declared software rule: every part is a plain rigid body. The arms are read through
rigstate.py (the let-go check).

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
from scoring import CAP_A, CAP_B, ROLL, TUBE  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Slide the rolled poster into the mailing tube, then press an end cap onto each end of the tube until "
               "it is seated.")


# -- the parts --------------------------------------------------------------------------------------------------------
def tube_shapes():
    """A ring of 16 flat kraft-board staves (1.5 mm), the bore 51 mm across the flats, along local y."""
    out = []
    r = L.TUBE_R_IN + L.TUBE_WALL / 2
    for k in range(L.TUBE_STAVES):
        th = 2 * math.pi * k / L.TUBE_STAVES
        out.append(mk.box((r * math.sin(th), 0.0, r * math.cos(th)), (L.TUBE_STAVE_W, L.TUBE_L, L.TUBE_WALL),
                          L.TUBE_COLOR, ypr=(0.0, th, 0.0)))
    return out


def cap_shapes():
    """Base frame at the centre of the flange's inner face; the flange at local y 0..5 mm, the plug (and its lead-in
    nose) at local y < 0; all along local y."""
    return [
        mk.cylinder((0.0, L.CAP_FLANGE_T / 2, 0.0), L.CAP_FLANGE_R, L.CAP_FLANGE_T, "Y", L.CAP_COLOR),
        mk.cylinder((0.0, -L.CAP_PLUG_L / 2, 0.0), L.CAP_PLUG_R, L.CAP_PLUG_L, "Y", L.CAP_COLOR),
        mk.cylinder((0.0, -L.CAP_PLUG_L - L.CAP_NOSE_L / 2, 0.0), L.CAP_NOSE_R, L.CAP_NOSE_L, "Y", L.CAP_COLOR),
    ]


def rest_shapes():
    """A cap rest, in the cap's frame turned to the table (z up from the table top): a platform under the plug and a
    rail each side of the flange."""
    out = [mk.box((0.0, -L.CAP_PLUG_L / 2, L.REST_PLAT_T / 2), (L.REST_PLAT_X, L.REST_PLAT_Y, L.REST_PLAT_T),
                  L.REST_COLOR)]
    for s in (-1, 1):
        out.append(mk.box((s * L.REST_RAIL_X, -0.005, L.REST_RAIL_H / 2),
                          (L.REST_RAIL_W, L.REST_RAIL_L, L.REST_RAIL_H), L.REST_COLOR))
    return out


BOARD = mk.assembly("board", origin=(L.BOARD_X, 0.0, L.TOP),
                    shapes=[mk.box((0.0, 0.0, L.BOARD_T / 2), (L.BOARD_LX, L.BOARD_LY, L.BOARD_T), L.BOARD_COLOR)],
                    color=L.BOARD_COLOR, friction=L.BOARD_FRICTION, tight_fit=True)
TUBE_PART = mk.assembly(TUBE, origin=(L.BOARD_X, 0.0, L.TOP + L.TUBE_Z), mass=L.TUBE_MASS, shapes=tube_shapes(),
                        color=L.TUBE_COLOR, friction=L.TUBE_FRICTION, tight_fit=True)
ROLL_PART = mk.assembly(ROLL, origin=(L.BOARD_X, L.ROLL_Y0, L.TOP + L.ROLL_Z0), mass=L.ROLL_MASS,
                        shapes=[mk.cylinder((0.0, 0.0, 0.0), L.ROLL_R, L.ROLL_L, "Y", L.ROLL_COLOR)],
                        color=L.ROLL_COLOR, friction=L.ROLL_FRICTION, tight_fit=True)


def cap_part(side: str):
    return mk.assembly(L.CAP_NAMES[side], origin=(L.CAP_X0, L.CAP_Y0[side], L.TOP + L.CAP_Z0), yaw=L.CAP_YAW[side],
                       mass=L.CAP_MASS, shapes=cap_shapes(), color=L.CAP_COLOR, friction=L.CAP_FRICTION,
                       tight_fit=True)


def rest_part(side: str):
    return mk.assembly(f"rest_{side}", origin=(L.CAP_X0, L.CAP_Y0[side], L.TOP), yaw=L.CAP_YAW[side],
                       shapes=rest_shapes(), color=L.REST_COLOR, tight_fit=True)


CAP_A_PART, CAP_B_PART = cap_part("a"), cap_part("b")
OBJECTS = [BOARD, rest_part("a"), rest_part("b"), TUBE_PART, ROLL_PART, CAP_A_PART, CAP_B_PART]
SPECS = {TUBE: TUBE_PART, ROLL: ROLL_PART, CAP_A: CAP_A_PART, CAP_B: CAP_B_PART}
OUT = {"a": 1.0, "b": -1.0}  # which way a cap's flange faces (world y) when it lies on its rest or sits on its end
_CAP_PART = {"a": CAP_A, "b": CAP_B}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it neither slides the roll into the
# tube nor lets a pressed cap push the roll or the tube. Offline only (never in the simulator), this model gives the
# scorer: (1) the roll and each cap carried by the gripper that is closed on it; (2) a released roll lying in the bore
# when its axis is in line with the tube's, else on the table; (3) a released cap staying on the tube's end when its
# plug is in the bore, else on its rest; (4) a held cap's nose pushing the roll's trailing end, and its flange pushing
# the tube, along the tube. No friction, no contact forces.
FULL_GAP = rigstate.FULL_GAP
HOLD_TOL = 0.012  # a gripper whose TCP is this close (x, z) to a part's centre line is on it
IN_BORE_XZ = 0.010
ROLL_X_TOL = 0.025  # the roll's sideways drift before the pads close on it, and how far the pads pull it in


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tube = np.array([L.BOARD_X, 0.0, L.TUBE_Z])
        self.roll = np.array([L.BOARD_X, L.ROLL_Y0, L.ROLL_Z0])
        self.cap = {s: np.array([L.CAP_X0, L.CAP_Y0[s], L.CAP_Z0]) for s in "ab"}
        self.grip: dict = {}  # part -> (arm index, offset of the part from the TCP)
        self.on_tube = {"roll": False, "a": False, "b": False}  # rides with the tube (rel y kept)
        self.rel = {"roll": 0.0, "a": 0.0, "b": 0.0}  # y from the tube's centre when it rides

    # -- helpers -------------------------------------------------------------------------------------------------------
    def _holder(self, read, name):
        """Which gripper (index) is closed on part ``name``: its pads past the part's width, its TCP on the part."""
        for i, (tcp, tips, opening) in enumerate(read):
            gap = float(opening) * FULL_GAP
            p = self.roll if name == "roll" else self.cap[name]
            if name in self.grip and self.grip[name][0] == i:  # already in this hand: it stays until the pads open
                width = 2 * L.ROLL_R if name == "roll" else 2 * L.CAP_FLANGE_R
                if gap <= width + 0.004:
                    return i
                continue
            if name == "roll":
                if gap > 2 * L.ROLL_R + 0.004 or abs(tcp[1] - p[1]) > L.ROLL_L / 2:
                    continue
                if abs(tcp[0] - p[0]) <= ROLL_X_TOL and abs((tcp[2] - L.TOP) - p[2]) <= HOLD_TOL:
                    return i  # the roll drifts sideways before the pickup (see ROLL_DRIFT_X): the pads centre it
                continue
            else:
                mid = p[1] + OUT[name] * L.CAP_FLANGE_T / 2
                if gap > 2 * L.CAP_FLANGE_R + 0.004 or abs(tcp[1] - mid) > HOLD_TOL:
                    continue
            if abs(tcp[0] - p[0]) <= HOLD_TOL and abs((tcp[2] - L.TOP) - p[2]) <= HOLD_TOL:
                return i
        return None

    def _in_line(self, p, tol=IN_BORE_XZ) -> bool:
        return abs(p[0] - self.tube[0]) <= tol and abs(p[2] - self.tube[2]) <= tol

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        # 1. what each gripper holds
        for name in ("roll", "a", "b"):
            i = self._holder(read, name)
            tcp = None if i is None else np.array([read[i][0][0], read[i][0][1], read[i][0][2] - L.TOP])
            part = self.roll if name == "roll" else self.cap[name]
            if i is None:
                if name in self.grip:
                    self._release(name)
                    self.grip.pop(name, None)
                continue
            if name not in self.grip or self.grip[name][0] != i:
                off = part - tcp
                if name == "roll":
                    off[0] = 0.0  # the pads centre the roll on the hand
                self.grip[name] = (i, off)
                self.on_tube[name] = False
            part[:] = tcp + self.grip[name][1]
        # 2. the tube's riders
        for name in ("roll", "a", "b"):
            if self.on_tube[name] and name not in self.grip:
                part = self.roll if name == "roll" else self.cap[name]
                part[0] = self.tube[0]
                part[1] = self.tube[1] + self.rel[name]
                part[2] = self.tube[2] - (L.TUBE_R_IN - L.ROLL_R if name == "roll" else 0.0005)
        # 3. a held cap whose plug is in line with the bore pushes the roll's trailing end and the tube
        for s in "ab":
            if s not in self.grip:
                continue
            c = self.cap[s]
            if not self._in_line(c, 0.004):
                continue
            out = OUT[s]
            end = self.tube[1] + out * L.TUBE_END
            tip = c[1] - out * L.CAP_DEPTH
            if self.on_tube["roll"] or "roll" not in self.grip:
                tr = self.roll[1] + out * L.ROLL_L / 2  # the roll's end nearest this cap
                if self._roll_in_bore() and out * (tip - tr) < 0.0:
                    self.roll[1] += tip - tr
                    self.rel["roll"] = self.roll[1] - self.tube[1]
            if out * (c[1] - end) < 0.0:  # the flange stops on the tube's end: the arm's over-travel is its lag
                c[1] = end
                i = self.grip[s][0]
                self.grip[s] = (i, c - np.array([read[i][0][0], read[i][0][1], read[i][0][2] - L.TOP]))

    def _roll_in_bore(self) -> bool:
        return self._in_line(self.roll, 0.012) and abs(self.roll[1] - self.tube[1]) < L.TUBE_END + L.ROLL_L / 2

    def _release(self, name) -> None:
        """A part let go of: it settles where it is supported."""
        if name == "roll":
            overlap = (self.tube[1] - L.TUBE_END + 0.005) < self.roll[1] + L.ROLL_L / 2 and \
                      self.roll[1] - L.ROLL_L / 2 < (self.tube[1] + L.TUBE_END - 0.005)
            if self._in_line(self.roll) and overlap:
                self.roll[0] = self.tube[0]
                self.roll[2] = self.tube[2] - (L.TUBE_R_IN - L.ROLL_R)
                self.on_tube["roll"] = True
                self.rel["roll"] = self.roll[1] - self.tube[1]
            else:
                self.roll[2] = L.ROLL_Z0
                self.on_tube["roll"] = False
            return
        c = self.cap[name]
        out = OUT[name]
        end = self.tube[1] + out * L.TUBE_END
        nose_in = out * (end - (c[1] - out * L.CAP_DEPTH)) > 0.002 and self._in_line(c, 0.004)
        if nose_in and out * (c[1] - end) < 0.03:
            c[0], c[2] = self.tube[0], self.tube[2] - 0.0005
            self.on_tube[name] = True
            self.rel[name] = c[1] - self.tube[1]
        else:
            c[2] = L.CAP_Z0
            self.on_tube[name] = False

    def states(self) -> dict:
        top = np.array([0.0, 0.0, L.TOP])
        ident = (0.0, 0.0, 0.0, 1.0)
        q_b = tuple(float(v) for v in mk.quat_from_ypr(math.pi, 0.0, 0.0))
        q_a = tuple(float(v) for v in mk.quat_from_ypr(0.0, 0.0, 0.0))
        return {
            TUBE: mk.state_from_poses(TUBE_PART, (*(self.tube + top), *ident), source="offline"),
            ROLL: mk.state_from_poses(ROLL_PART, (*(self.roll + top), *ident), source="offline"),
            CAP_A: mk.state_from_poses(CAP_A_PART, (*(self.cap["a"] + top), *q_a), source="offline"),
            CAP_B: mk.state_from_poses(CAP_B_PART, (*(self.cap["b"] + top), *q_b), source="offline"),
        }


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(TUBE), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, roll ends mm, cap gaps mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: every part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The roll in the tube, a seated cap on each end, both grippers 30 mm clear and every part still for 1 s, no
    latched loss; held 1 s. This runs first every control step, so it also feeds the tracker (and, without a
    simulator, the offline stand-in)."""
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
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(TUBE), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["roll_end_s_mm"], d["cap_a_axial_mm"], d["cap_b_axial_mm"],
                      d["cap_a_radial_mm"], d["cap_b_radial_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the roll is in the tube and every part is let go and still, and nothing was lost; then 1.0 with a
    seated cap on both ends, 0.5 with one, 0.25 with none. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
OPEN_ROLL = -(2 * L.ROLL_R + 0.028) / 2  # 66 mm round the 38 mm roll
GRIP_ROLL = -(2 * L.ROLL_R - 0.008) / 2  # 30 mm commanded across the roll
OPEN_CAP = -(2 * L.CAP_FLANGE_R + 0.016) / 2  # 76 mm round the 60 mm flange
GRIP_CAP = -(2 * L.CAP_FLANGE_R - 0.008) / 2  # 52 mm commanded across the flange
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # both close along x
YAW_ROLL = math.pi / 2  # the left hand at the roll's far end, 0.30 m outboard of its base: joint 6 needs this sign there

HOVER = 0.120
# the left hand takes the roll twice. A pinch holds no torque about its closing axis (x), so a roll held off its middle
# swings about the pads: the GPU run held it 25 mm from its far end and its near end hung on the table, dragged into the
# mouth's ring and shoved the tube. First hold: the middle (balanced, it carries level) to bring the near half in. Second
# hold: the far end, only to push the roll along the bore floor, where the bore carries it.
ROLL_MID_Y = L.ROLL_Y0  # 0.435: the roll's middle
ROLL_DRIFT_X = 0.282  # the roll rolls sideways on the table at about 4 mm/s from the first step (GPU settle: 32 mm in
# 8 s, 17 mm by the time the pads arrive, toward -x): the hand comes down where the roll will be, not where it started
ROLL_CARRY_Z = L.TUBE_Z  # the roll's axis on the tube's axis (6.5 mm of play every way), lifted 13 mm commanded
ROLL_END_Y = L.TUBE_END + 0.005 + rigstate.FINGER_HALF_WIDTH  # the pads' near edge 5 mm outside the mouth: 0.247
ROLL_SPEED = 0.040  # m/s
ROLL_TAIL_IN = 0.025  # second hold: this far in from the far end
ROLL_PUSH_Z = 0.027  # second hold, during the push: the far end rides on the bore's edge (axis 25.5 mm at the mouth)
# the caps: lifted clear of the rails, carried in line, lowered to the tube's axis, pressed on
CAP_GRIP_Z = L.CAP_Z0
CAP_CARRY_Z = 0.050  # cap axis 20 mm up: the flange clears the 12 mm rails
CAP_PRESS_Z = L.TUBE_Z + 0.001
CAP_FLANGE_MID = L.CAP_FLANGE_T / 2
PRESS_SPEED = 0.025
OVERTRAVEL = 0.002  # commanded past the seat: the arm trails its command


def cap_y(side: str, gap: float) -> float:
    """TCP y of the flange's middle when the flange's inner face is ``gap`` outside its tube end."""
    return OUT[side] * (L.TUBE_END + gap + CAP_FLANGE_MID)


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def roll_moves():
    """The left arm, first hold: down round the roll's middle, grip it, lift it 13 mm to the tube's axis, slide it in end
    first along y until the pads are 5 mm outside the mouth, let go, up. Second hold: over the roll's far end (it now
    sticks 208 mm out), grip it, lift it a little, push it in along the bore floor until the pads are 5 mm outside the
    mouth again (the roll's far end 43 mm out), let go, up."""
    x, y0, y1 = L.BOARD_X, ROLL_MID_Y, ROLL_END_Y
    xd = ROLL_DRIFT_X
    y_tail = y1 + L.ROLL_L / 2 - ROLL_TAIL_IN  # the middle is at y1 after the first slide: the far end is half a roll out
    return [
        _w(2.0, xd, y0, HOVER, OPEN_ROLL, "over the roll's middle", YAW_ROLL),
        _w(1.5, xd, y0, 0.050, OPEN_ROLL, "down to just above it", YAW_ROLL),
        _w(1.5, xd, y0, L.ROLL_Z0, OPEN_ROLL, "slowly down round it", YAW_ROLL),
        _w(0.5, xd, y0, L.ROLL_Z0, OPEN_ROLL, "settle", YAW_ROLL),
        _w(1.5, xd, y0, L.ROLL_Z0, GRIP_ROLL, "grip the roll", YAW_ROLL),
        _w(1.5, x, y0, ROLL_CARRY_Z, GRIP_ROLL, "lift it to the tube's axis", YAW_ROLL),
        _w((y0 - y1) / ROLL_SPEED, x, y1, ROLL_CARRY_Z, GRIP_ROLL, "slide the roll into the tube", YAW_ROLL),
        _w(0.5, x, y1, ROLL_CARRY_Z, GRIP_ROLL, "settle", YAW_ROLL),
        _w(1.5, x, y1, ROLL_CARRY_Z, OPEN_ROLL, "let go of the roll", YAW_ROLL),
        _w(1.0, x, y1, HOVER, OPEN_ROLL, "straight up", YAW_ROLL),
        _w(1.5, x, y_tail, HOVER, OPEN_ROLL, "over the roll's far end", YAW_ROLL),
        _w(1.5, x, y_tail, 0.050, OPEN_ROLL, "down to just above it", YAW_ROLL),
        _w(1.5, x, y_tail, L.ROLL_Z0 + 0.001, OPEN_ROLL, "slowly down round it", YAW_ROLL),
        _w(0.5, x, y_tail, L.ROLL_Z0 + 0.001, OPEN_ROLL, "settle", YAW_ROLL),
        _w(1.5, x, y_tail, L.ROLL_Z0 + 0.001, GRIP_ROLL, "grip the far end", YAW_ROLL),
        _w(1.0, x, y_tail, ROLL_PUSH_Z, GRIP_ROLL, "lift it a little", YAW_ROLL),
        _w((y_tail - y1) / ROLL_SPEED, x, y1, ROLL_PUSH_Z, GRIP_ROLL, "push the roll along the bore", YAW_ROLL),
        _w(0.5, x, y1, ROLL_PUSH_Z, GRIP_ROLL, "settle", YAW_ROLL),
        _w(1.5, x, y1, ROLL_PUSH_Z, OPEN_ROLL, "let go of the roll", YAW_ROLL),
        _w(1.0, x, y1, HOVER, OPEN_ROLL, "straight up", YAW_ROLL),
    ]


def cap_fetch(side: str):
    """One arm: over its cap, down round the flange, grip it, lift it clear of the rails, carry it in line with its
    end of the tube and lower it to the tube's axis, 70 mm from the mouth."""
    arm_yaw = YAW_L if side == "a" else YAW_R
    y_rest = OUT[side] * (abs(L.CAP_Y0[side]) + CAP_FLANGE_MID)
    y_stage = OUT[side] * (abs(L.CAP_Y0[side]) + CAP_FLANGE_MID)
    return [
        _w(2.0, L.CAP_X0, y_rest, HOVER, OPEN_CAP, f"over cap {side} on its rest", arm_yaw),
        _w(1.0, L.CAP_X0, y_rest, 0.060, OPEN_CAP, "down to just above it", arm_yaw),
        _w(1.5, L.CAP_X0, y_rest, CAP_GRIP_Z, OPEN_CAP, "slowly down round the flange", arm_yaw),
        _w(0.5, L.CAP_X0, y_rest, CAP_GRIP_Z, OPEN_CAP, "settle", arm_yaw),
        _w(1.5, L.CAP_X0, y_rest, CAP_GRIP_Z, GRIP_CAP, "grip the flange", arm_yaw),
        _w(1.5, L.CAP_X0, y_rest, CAP_CARRY_Z, GRIP_CAP, "lift it clear of the rails", arm_yaw),
        _w(2.0, L.BOARD_X, y_stage, CAP_CARRY_Z, GRIP_CAP, "carry it in line with the tube", arm_yaw),
        _w(1.0, L.BOARD_X, y_stage, CAP_PRESS_Z, GRIP_CAP, "lower it to the tube's axis", arm_yaw),
        _w(0.5, L.BOARD_X, y_stage, CAP_PRESS_Z, GRIP_CAP, "settle", arm_yaw),
    ]


def cap_press(side: str, gap_left: float = 0.0):
    """Press the cap on: the flange's inner face from 100 mm out to the tube's end (plus the over-travel), settle, let
    go, up, aside. ``gap_left``: stop this far short (counterfactuals)."""
    arm_yaw = YAW_L if side == "a" else YAW_R
    y_stage = cap_y(side, abs(L.CAP_Y0[side]) - L.TUBE_END)
    y_end = cap_y(side, gap_left) - OUT[side] * OVERTRAVEL
    park = (0.20, OUT[side] * 0.40, 0.16)
    return [
        _w(abs(y_stage - y_end) / PRESS_SPEED, L.BOARD_X, y_end, CAP_PRESS_Z, GRIP_CAP, f"press cap {side} on", arm_yaw),
        _w(0.5, L.BOARD_X, y_end, CAP_PRESS_Z, GRIP_CAP, "settle", arm_yaw),
        _w(1.5, L.BOARD_X, y_end, CAP_PRESS_Z, OPEN_CAP, "let go", arm_yaw),
        _w(1.0, L.BOARD_X, y_end, HOVER, OPEN_CAP, "straight up", arm_yaw),
        _w(1.5, park[0], park[1], park[2], OPEN, "aside", arm_yaw),
    ]


# Wrong ways (counterfactuals): how far short of its seat each cap is pressed.
VARIANT_GAP = {"short": (0.010, 0.010), "right_short": (0.0, 0.010)}  # (cap a, cap b) gap left in metres
VARIANTS = tuple(VARIANT_GAP)


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the left arm slides the roll in (two holds); meanwhile the right arm waits over cap b; the two
    arms then pick cap a and cap b together and bring them in line with the tube's ends; both caps are pressed
    on at the same time (the tube is free on its board; the roll's trailing end is pushed home by cap a)."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    gap_a, gap_b = VARIANT_GAP.get(variant, (0.0, 0.0))
    left = roll_moves() + cap_fetch("a")
    right = cap_fetch("b")
    wait = _duration(left) - _duration(right)
    if wait > 0.0:  # the right hand waits over its cap, not holding it: it fetches while the left hand fetches cap a
        right.insert(1, _w(wait, L.CAP_X0, OUT["b"] * (abs(L.CAP_Y0["b"]) + CAP_FLANGE_MID), HOVER, OPEN_CAP,
                           "wait over cap b while the roll goes in", YAW_R))
    left += cap_press("a", gap_a)
    right += cap_press("b", gap_b)
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short        both caps stopped 10 mm short of their seats: the roll's end only 8 mm in (0.0)
      right_short  the right cap 10 mm short: left cap seated (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
