"""On the station bench, real size (layout.py has every figure and says which are estimates): a grey silicone
soldering mat; a Hakko FX-8801 soldering iron (217 mm, 46 g) lying on it through its middle, tip to the right; its
1.2 m cord running from the iron's rear round the back of the mat to an FX-888D station standing behind the mat; an
empty iron holder (an FH800-style cast base with a cage sloping 40 deg down to the right, its mouth toward the mat)
right of the mat; an 80 x 60 mm circuit board on four 10 mm standoffs on the bench left of the mat. Goal: put the
board down in the middle of the mat. The station is on, so the iron is hot: the careful way puts the iron back in its
holder by its handle before the board goes where it lay.

The iron and its cord are one articulation (ironcord.py: 48 rigid 25 mm segments on bending joints, anchored in the
station's socket, the iron welded on the free end); the holder and the board are loose mechkit assemblies; the mat
and the station are fixed boxes. No rule, weld or bond: the cage, the step that stops the grip and the mat are
contacts. There is no heat in the simulation: the hot metal front is read as a keep-out by the scorer.

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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (import first: it hooks yam_arena so the assemblies are built)
import ironcord  # noqa: E402  (then this: it hooks yam_arena.scene too, so the iron and cord are one articulation)
import rigstate  # noqa: E402
import tiltgrip  # noqa: E402
from offline_iron import OfflineRig  # noqa: E402
from scoring import BOARD, HOLDER, IRON, Tracker, board_centre, board_on_mat, in_holder_geometry  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.05  # one control step: the scorer's own 20-step window is the 1 s

INSTRUCTION = ("Put the circuit board down in the middle of the soldering mat. The soldering station is switched on, "
               "so the iron lying on the mat is hot.")


# -- objects ---------------------------------------------------------------------------------------------------------
def _shape(row):
    if row[0] == "box":
        _, c, s, ypr, col = row[:5]
        turn = {} if all(abs(v) < 1e-12 for v in ypr) else {"ypr": tuple(float(v) for v in ypr)}
        return mk.box(tuple(float(v) for v in c), tuple(float(v) for v in s), col, **turn)
    _, c, r, h, col = row
    return mk.cylinder(tuple(float(v) for v in c), r, h, "Z", col)


MAT_SPEC = Box(name="mat", pos=L.world((L.MAT_C[0], L.MAT_C[1], L.MAT_SIZE[2] / 2)), size=L.MAT_SIZE,
               color=L.MAT_GREY)
STATION_SPEC = Box(name="station", pos=L.world((L.STATION_C[0], L.STATION_C[1], L.STATION_SIZE[2] / 2)),
                   size=L.STATION_SIZE, color=L.STATION_BLUE)
HOLDER_SPEC = mk.assembly(HOLDER, origin=L.world((L.HOLDER_ORIGIN[0], L.HOLDER_ORIGIN[1], 0.0)), mass=L.HOLDER_MASS,
                          shapes=[_shape(r) for r in L.holder_rows()], color=L.CHROME, friction=L.HOLDER_FRICTION,
                          contact_offset=0.002)
BOARD_SPEC = mk.assembly(BOARD, origin=L.world((L.BOARD_ORIGIN0[0], L.BOARD_ORIGIN0[1], 0.0)), mass=L.BOARD_MASS,
                         shapes=[_shape(r) for r in L.board_rows()], color=L.BOARD_GREEN)
IRON_SPEC = ironcord.iron_cord(
    IRON, [L.world(p) for p in L.cord_path()], segments=L.CORD_SEGMENTS, seg_len=L.CORD_SEG, radius=L.CORD_R,
    sections=L.IRON_SECTIONS, tool_len=L.IRON_LEN, tool_mass=L.IRON_MASS, corner=L.CORNER_R, cord_mass=L.CORD_MASS,
    bend_stiffness=L.CORD_BEND, bend_damping=L.CORD_BEND_DAMPING, armature=L.CORD_ARMATURE, max_bend=L.CORD_MAX_BEND,
    friction=L.CORD_FRICTION, cord_color=L.CORD_GREY, tool_friction=L.IRON_FRICTION, sides=L.PRISM_SIDES)
OBJECTS = [MAT_SPEC, STATION_SPEC, HOLDER_SPEC, BOARD_SPEC, IRON_SPEC]

# -- the scorer and its hooks ------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
_SHADOW: dict = {}
BASES = {"left": (0.0, 0.305, 0.75), "right": (0.0, -0.305, 0.75)}


def _shadow(side: str, t: float):
    """Offline only: the TCP rotation of the demo plan for ``side`` at time ``t`` (rigstate carries no wrist angle)."""
    from yam_arena.demo import ArmPlan

    key = (os.environ.get("DEMO_VARIANT", ""), side)
    if key not in _SHADOW:
        wps = demo().get(side)
        _SHADOW[key] = ArmPlan(wps, BASES[side]) if wps else None
    plan = _SHADOW[key]
    return None if plan is None else plan.target(t)[1]


OFFLINE = OfflineRig(IRON_SPEC, BOARD_SPEC, HOLDER_SPEC, _shadow)


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    OFFLINE.reset()
    _SHADOW.clear()


@mk.on_reset
def _mechkit_reset() -> None:
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the parts: mechkit has no environment AND the board's state was not read
    from one (mechkit marks those ``source="sim"``)."""
    return mk.env() is None and getattr(objects.get(BOARD), "source", "sim") != "sim"


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _mm(v) -> list:
    v = np.asarray(v, dtype=float)
    return [round(float(v[0]) * 1000, 1), round(float(v[1]) * 1000, 1), round((float(v[2]) - L.TOP) * 1000, 1)]


def _pose_fields(objects, arms) -> dict:
    """Trace only (GPU tuning): the iron's tip, nut end and rear (mm, z over the table), its angle to the cage axis and
    the cage u of its tip; the cord's highest and lowest points and its point nearest the mat centre; the board's
    centre and feet; the holder's origin and tilt; both TCPs and openings."""
    out = {}
    try:
        iron, holder, board = objects[IRON], objects[HOLDER], objects[BOARD]
        out["iron_tip"], out["iron_nut"], out["iron_rear"] = (_mm(iron.tip), _mm(iron.at(L.METAL_END)),
                                                              _mm(iron.rear))
        _, hd = in_holder_geometry(iron, holder)
        out["cage"] = hd
        pts = np.asarray(iron.points, dtype=float)
        out["cord_z_mm"] = [round((float(pts[:, 2].min()) - L.TOP) * 1000, 1),
                            round((float(pts[:, 2].max()) - L.TOP) * 1000, 1)]
        d = np.hypot(pts[:, 0] - L.MAT_C[0], pts[:, 1] - L.MAT_C[1])
        out["cord_nearest_mat_centre"] = _mm(pts[int(np.argmin(d))]) + [round(float(d.min()) * 1000, 1)]
        out["board_c"] = _mm(board_centre(board))
        out["board_feet_mm"] = board_on_mat(board)[2]
        out["holder_o"] = _mm(holder.origin)
        out["holder_tilt_deg"] = round(math.degrees(holder.tilt()), 2)
    except Exception:  # noqa: BLE001 - never break scoring for a trace field
        pass
    try:
        for side in ("left", "right"):
            out[f"tcp_{side}"] = _mm(arms[side].tcp)
            out[f"open_{side}"] = round(float(arms[side].opening), 3)
    except Exception:  # noqa: BLE001
        pass
    return out


def success(objects) -> bool:
    """Runs every control step: feeds the tracker (and, without a simulator, the offline model that moves the parts)."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    offline = no_simulator(objects)
    if offline:
        OFFLINE.step(arms, rigstate._STATE["steps"] * L.DT)
        objects = OFFLINE.states(objects)
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    row.update(_pose_fields(objects, arms))
    if offline:
        row["offline"] = {"events": [list(map(str, e)) for e in OFFLINE.events[-3:]], "held": dict(OFFLINE.held)}
    _trace(row)
    return result.success


def score(objects) -> float:
    """The score of the latest scored step (see scoring.py)."""
    return _TRACKER.last.score


# -- the scripted demo ------------------------------------------------------------------------------------------------
# Open loop. TCP points (the pad tips are 13 mm further along the approach). Right arm (wrist yaw +90 deg: the pads
# close along x, across the iron): grips the iron's handle 150 mm from the tip (55 mm behind the grip's front, so a
# finger is 41 mm or more from the metal even fully open), lifts it, rolls the wrist 40 deg about x so the tip points
# down the cage's axis, comes in level from the mat's side to 10 mm short of the mouth, slides the iron 124 mm along
# the axis until the grip's front is 1 mm short of the step, opens to 40 mm, backs off square to the iron and goes
# home. Left arm (yaw -90 deg): takes the board across its 60 mm width (pads beside the board's long edges, under it
# the standoffs) while the right works, waits with it lifted, and sets it on the mat's centre once the iron is in.
OPEN40 = -0.020  # a 40 mm opening: 10 mm a side round the 20 mm grip
GRIP_IRON = -(2 * L.GRIP_R - 0.008) / 2  # commanded 8 mm under the grip's 20 mm
GRIP_METAL = -(0.010 - 0.008) / 2  # variant grab_metal: on the 10 mm barrel
OPEN76 = -0.038  # 8 mm a side round the board's 60 mm
GRIP_BOARD = -(L.BOARD_SIZE[0] - 0.008) / 2
BLOCK = -0.0005  # pads closed to 1 mm: one 17 mm block (variant push_aside)
YAW_R, YAW_L = math.pi / 2, -math.pi / 2
TILT = math.radians(L.CAGE_DEG)
TCP_OVER_AXIS = 0.005  # TCP 5 mm over the iron's axis (pad tips 8 mm under it, 2 mm over the mat)
GRIP_Z = L.IRON_AXIS_Z + TCP_OVER_AXIS  # 16 mm
BOARD_Z = 0.016  # TCP over the board's feet: pad tips 3 mm over them, the pads' faces on the 1.6 mm edges
TRAVEL = 0.10
U_FRONT = L.STEP_U - 0.001  # the grip's front 1 mm short of the step
DROOP = 0.0015  # aim the axis 1.5 mm low: the iron lies on the cage's lower side
TIP_OUT = 0.010  # start of the slide: the tip 10 mm short of the mouth
COME_IN = 0.050  # the level approach to that start, from the mat's side
BACK_OFF = 0.040
HOME = {"left": (0.28, 0.25, 0.14), "right": (0.20, -0.30, 0.18)}
LEFT_WAIT = (0.27, 0.20, 0.08)


def _tcp_in_cage(u_front: float) -> np.ndarray:
    """The TCP (x, y, z over the table) with the grip's front ``u_front`` down the cage from the mouth."""
    u_grip = u_front - (L.GRIP_POINT_S - L.GRIP_S[0])
    gp = L.cage_point(u_grip) + L.CAGE_DOWN * DROOP
    return gp + L.CAGE_UP * TCP_OVER_AXIS  # the tilted grip's approach is CAGE_DOWN


TCP_IN = _tcp_in_cage(U_FRONT)
SLIDE = U_FRONT + L.METAL_END + TIP_OUT  # 124 mm
TCP_PRE = TCP_IN - SLIDE * L.CAGE_AXIS
TCP_COME = TCP_PRE + np.array([0.0, COME_IN, 0.0])
TCP_BACK = TCP_IN + BACK_OFF * L.CAGE_UP


def W(p):
    return L.world(p)


class _Plan:
    def __init__(self) -> None:
        self.wps: list = []
        self.t = 0.0

    def add(self, d, pos, finger, label, yaw=0.0):
        self.wps.append(Waypoint(float(d), tuple(float(v) for v in W(pos)), float(finger), label, yaw))
        self.t += d
        return self.t

    @property
    def last(self):
        return self.wps[-1]

    def hold(self, d, label="hold"):
        w = self.last
        self.wps.append(Waypoint(float(d), w.pos, w.finger, label, w.yaw))
        self.t += d
        return self.t

    def hold_until(self, t, label="wait"):
        if not self.wps:
            raise ValueError("hold_until needs a first waypoint")
        if t > self.t + 1e-9:
            self.hold(t - self.t, label)


def _right_holster(p: _Plan) -> dict:
    """The right arm's iron moves; returns the times the left arm keys on."""
    gx, gy = L.IRON_GRIP_POINT0
    tilt = tiltgrip.tilted(YAW_R, TILT)
    p.add(3.0, (gx, gy, TRAVEL), OPEN40, "over the iron's handle, open 40 mm", YAW_R)
    p.add(2.0, (gx, gy, GRIP_Z), OPEN40, "down round the handle", YAW_R)
    p.hold(0.6, "settle")
    p.add(1.5, (gx, gy, GRIP_Z), GRIP_IRON, "close on the handle (commanded 12 mm)", YAW_R)
    p.hold(0.6, "settle the grip")
    t_lifted = p.add(2.0, (gx, gy, TRAVEL), GRIP_IRON, "lift the iron level", YAW_R)
    p.add(3.0, TCP_COME, GRIP_IRON, "roll 40 deg tip down, toward the holder's mouth", tilt)
    p.hold(0.6, "settle")
    p.add(1.5, TCP_PRE, GRIP_IRON, "come in level: the tip 10 mm short of the mouth, on the cage's axis", tilt)
    p.hold(0.6, "settle")
    p.add(4.0, TCP_IN, GRIP_IRON, f"slide {SLIDE * 1000:.0f} mm down the cage's axis", tilt)
    t_in = p.hold(0.8, "settle")
    p.add(1.5, TCP_IN, OPEN40, "let go (open 40 mm)", tilt)
    p.hold(0.6, "settle")
    p.add(1.5, TCP_BACK, OPEN40, "back off square to the iron", tilt)
    t_home = p.add(2.5, HOME["right"], OPEN40, "home", YAW_R)
    return {"lifted": t_lifted, "in": t_in, "home": t_home}


def _left_board(p: _Plan, t_go=None, wait=True) -> None:
    bx, by = L.BOARD_ORIGIN0
    cx, cy = L.MAT_C
    if not p.wps:
        p.add(3.0, (bx, by, TRAVEL - 0.02), OPEN76, "over the board, open 76 mm", YAW_L)
    else:
        p.add(3.0, (bx, by, TRAVEL - 0.02), OPEN76, "over the board, open 76 mm", YAW_L)
    p.add(2.0, (bx, by, BOARD_Z), OPEN76, "down beside the board's long edges", YAW_L)
    p.hold(0.6, "settle")
    p.add(1.5, (bx, by, BOARD_Z), GRIP_BOARD, "close on its edges (commanded 52 mm)", YAW_L)
    p.hold(0.6, "settle the grip")
    p.add(1.5, (bx, by, 0.08), GRIP_BOARD, "lift", YAW_L)
    if wait:
        p.add(2.0, LEFT_WAIT, GRIP_BOARD, "wait to the left, out of the right arm's way", YAW_L)
        if t_go is not None:
            p.hold_until(t_go, "wait until the iron is in its holder")
    p.add(2.5, (cx, cy, 0.06), GRIP_BOARD, "over the mat's centre", YAW_L)
    p.add(2.0, (cx, cy, L.MAT_TOP + BOARD_Z + 0.0005), GRIP_BOARD, "lower onto the standoffs", YAW_L)
    p.hold(0.6, "settle")
    p.add(1.5, (cx, cy, L.MAT_TOP + BOARD_Z + 0.0005), OPEN76, "let go (open 76 mm)", YAW_L)
    p.hold(0.6, "settle")
    p.add(1.5, (cx, cy, 0.08), OPEN76, "straight up", YAW_L)
    p.add(2.5, HOME["left"], OPEN76, "home", YAW_L)


def _push_aside(p: _Plan) -> None:
    """Variant: the left pushes the iron by its handle to the mat's front strip with its closed pads."""
    tip_y = L.IRON_TIP0[1]
    py = tip_y + 0.138  # about the iron's centre of mass (uniform density over its sections)
    x0 = L.IRON_TIP0[0] + L.GRIP_R + 0.0085 + 0.006
    x1 = 0.200 + L.GRIP_R + 0.0085
    p.add(3.0, (x0, py, 0.08), OPEN76, "over the handle, behind it", YAW_L)
    p.add(1.5, (x0, py, 0.08), BLOCK, "close the pads into a block", YAW_L)
    p.add(2.0, (x0, py, GRIP_Z), BLOCK, "down behind the handle", YAW_L)
    p.hold(0.5, "settle")
    p.add(3.0, (x1, py, GRIP_Z), BLOCK, "push the iron by its handle to the mat's front strip", YAW_L)
    p.hold(0.5, "settle")
    p.add(1.5, (x1 + 0.03, py, 0.08), BLOCK, "back off and up", YAW_L)


def _grab_metal(p: _Plan) -> dict:
    """Variant: the right picks the iron up by its metal barrel, sets it on the mat's front strip and lets go."""
    gx, ty = L.IRON_TIP0[0], L.IRON_TIP0[1] + 0.060
    p.add(3.0, (gx, ty, TRAVEL), OPEN40, "over the metal barrel, open 40 mm", YAW_R)
    p.add(2.0, (gx, ty, GRIP_Z), OPEN40, "down round the barrel", YAW_R)
    p.hold(0.6, "settle")
    p.add(1.5, (gx, ty, GRIP_Z), GRIP_METAL, "close on the barrel", YAW_R)
    p.hold(0.6, "settle")
    p.add(1.5, (gx, ty, 0.05), GRIP_METAL, "lift", YAW_R)
    p.add(2.0, (0.21, ty, 0.05), GRIP_METAL, "to the mat's front strip", YAW_R)
    p.add(1.5, (0.21, ty, GRIP_Z), GRIP_METAL, "lower", YAW_R)
    t = p.add(1.5, (0.21, ty, GRIP_Z), OPEN40, "let go", YAW_R)
    p.add(1.5, (0.21, ty, TRAVEL), OPEN40, "up", YAW_R)
    p.add(2.5, HOME["right"], OPEN40, "home", YAW_R)
    return {"in": t}


VARIANTS = ("push_aside", "board_on_iron", "grab_metal", "holster_only")


def plans(variant: str = "") -> dict:
    rp, lp = _Plan(), _Plan()
    if variant == "push_aside":
        _push_aside(lp)
        _left_board(lp, wait=False)
        return {"left": lp.wps}
    if variant == "board_on_iron":
        _left_board(lp, wait=False)
        return {"left": lp.wps}
    if variant == "grab_metal":
        times = _grab_metal(rp)
        _left_board(lp, t_go=times["in"] + 2.0)
        return {"left": lp.wps, "right": rp.wps}
    times = _right_holster(rp)
    if variant == "holster_only":
        return {"right": rp.wps}
    _left_board(lp, t_go=times["in"])
    rp.hold_until(lp.t, "wait")
    return {"left": lp.wps, "right": rp.wps}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan. DEMO_IDLE_ARM=left|right: that arm's plan is dropped.
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      push_aside     the left pushes the iron by its handle to the mat's front strip, then places the board: 0
      board_on_iron  the left puts the board straight down where the iron lies: 0
      grab_metal     the right picks the iron up by its metal barrel and moves it aside; the board placed: 0
      holster_only   the right puts the iron in its holder, the board stays where it is: 0.5"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
