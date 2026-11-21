"""On the station table, real size: a utility alcove with a laundry tub's two supply stubs (the "cabinet"). A 1/2 in
plywood back panel stands 400 mm in front of the arm bases (300 mm high, 876 mm wide) with a side panel at each end
(inner faces at y +-420 mm). Through the back panel come two 1/2 in copper supply stubs (15.9 mm OD), 70 mm above the
floor, 203 mm (8 in) apart, each held at the panel by a chrome escutcheon and by two pipe clips on standoff posts. On
each stub, 130 mm from the panel, sits a 1/2 in brass full-port ball valve (body 52 mm end to end, 30 mm across) with
a stop plate and a flat steel lever 130 mm from the stem's axis to its tip, 18 x 6 mm, with a 60 mm vinyl grip over
its outer end. The left valve (y +101.5 mm) is the cold supply, blue grip; the right one (y -101.5 mm) is the hot
supply, red grip. Both levers start open, pointing along the pipe toward the robot.

Goal: shut the cold valve: swing its lever a quarter turn clockwise (seen from above) to its stop, so it stands across
the pipe toward +y; leave the hot lever alone; let go. The valve turns against 0.20 N m of friction (2.0 N at the
middle of the grip, est).

The cabinet is ONE mechkit assembly (mechkit.py, next to this file): the fixed frame (panels, stubs, valve bodies,
clips) with the two levers as hinges on the stems (axis -Z, so a clockwise turn is positive; limits 0 and 90 degrees:
the stop plate's two tabs; joint friction 0.20 N m). Nothing is moved by code. The arms are read through rigstate.py
(the let-go check).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the cabinet is built as an articulation)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import CABINET, COLD, HOT  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Shut off the cold supply at its blue-gripped ball valve, leaving the red-gripped hot valve open, and "
               "let go of it.")


# -- the parts --------------------------------------------------------------------------------------------------------
def _box(cx, cy, cz, sx, sy, sz, color):
    return mk.box((cx, cy, cz), (sx, sy, sz), color)


def _cyl(cx, cy, cz, radius, height, axis, color):
    return mk.cylinder((cx, cy, cz), radius, height, axis, color)


def valve_shapes(y: float) -> list:
    """One stub with its escutcheon, clips and valve body, fixed (base frame = the world with z above the table)."""
    z = L.PIPE_Z
    out = [
        _cyl((L.PIPE_X0 + L.BACK_X0) / 2, y, z, L.PIPE_R, L.BACK_X0 - L.PIPE_X0, "X", L.COPPER),
        _cyl(L.BACK_X0 - 0.002, y, z, 0.026, 0.004, "X", L.CHROME),  # escutcheon on the panel's face
        _cyl(L.STEM_X, y, z, L.BODY_R, L.HOUSING_L, "X", L.BRASS),  # the ball housing
        _cyl(L.STEM_X - (L.HOUSING_L + L.END_L) / 2, y, z, L.END_R, L.END_L, "X", L.BRASS),
        _cyl(L.STEM_X + (L.HOUSING_L + L.END_L) / 2, y, z, L.END_R, L.END_L, "X", L.BRASS),
        _cyl(L.STEM_X, y, (L.NECK_Z0 + L.PLATE_Z) / 2, L.NECK_R, L.PLATE_Z - L.NECK_Z0, "Z", L.BRASS),  # neck
        _box(L.STEM_X, y, L.PLATE_Z, L.PLATE_SIZE, L.PLATE_SIZE, 0.0015, L.CHROME),  # the stop plate
        # the two stop tabs: beside the open lever's -y face (it lies along -x) and the closed lever's +x face
        _box(L.STEM_X - 0.0135, y - 0.011, L.PLATE_Z + 0.00075 + L.TAB[2] / 2, *L.TAB, L.CHROME),
        _box(L.STEM_X + 0.011, y + 0.0135, L.PLATE_Z + 0.00075 + L.TAB[2] / 2, *L.TAB, L.CHROME),
    ]
    for cx in L.CLIP_X:  # a saddle clip: a strap over the pipe, two legs, on a standoff post from the floor
        top = z + L.PIPE_R
        out += [
            _box(cx, y, (z - L.PIPE_R) / 2, L.CLIP_POST[0], L.CLIP_POST[1], z - L.PIPE_R, L.WOOD),
            _box(cx, y, top + 0.00125, 0.012, 2 * L.PIPE_R + 0.005, 0.0025, L.CLIP_PLASTIC),
            _box(cx, y - L.PIPE_R - 0.00125, z, 0.012, 0.0025, 2 * L.PIPE_R, L.CLIP_PLASTIC),
            _box(cx, y + L.PIPE_R + 0.00125, z, 0.012, 0.0025, 2 * L.PIPE_R, L.CLIP_PLASTIC),
        ]
    return out


def frame_shapes() -> list:
    side_x = (L.SIDE_X0 + L.BACK_X1) / 2
    out = [_box(L.BACK_X0 + L.BACK_T / 2, 0.0, L.BACK_H / 2, L.BACK_T, 2 * L.BACK_HALF_Y, L.BACK_H, L.WOOD)]
    for s in (1, -1):
        out.append(_box(side_x, s * (L.SIDE_IN_Y + L.SIDE_T / 2), L.SIDE_H / 2, L.BACK_X1 - L.SIDE_X0, L.SIDE_T,
                        L.SIDE_H, L.WOOD))
    return out + valve_shapes(L.COLD_Y) + valve_shapes(L.HOT_Y)


def lever_shapes(y: float, grip_color) -> list:
    """The lever at its open pose: the steel bar from 12 mm behind the stem to the tip 130 mm toward -x, the vinyl
    grip over its outer 60 mm, the stem nut on top."""
    x0 = L.STEM_X + L.BAR_TAIL
    x1 = L.STEM_X - L.BAR_L
    return [
        _box((x0 + x1) / 2, y, L.LEVER_Z, x0 - x1, L.BAR_W, L.BAR_T, L.STEEL),
        _cyl(L.STEM_X - L.GRIP_C, y, L.LEVER_Z, L.GRIP_R, L.GRIP_L, "X", grip_color),
        _cyl(L.STEM_X, y, L.LEVER_Z + L.BAR_T / 2 + 0.003, 0.007, 0.006, "Z", L.BRASS),
    ]


def _lever(name: str, y: float, grip_color):
    return mk.hinge(name, "-Z", (L.STEM_X, y, L.LEVER_Z), lever_shapes(y, grip_color), lower=0.0, upper=L.Q_CLOSED,
                    mass=L.LEVER_MASS, damping=L.LEVER_DAMPING, friction=L.VALVE_FRICTION_TORQUE)


class Cabinet(mk.Assembly):
    """The fixed frame, as the offline checker draws it: the back panel's box, so the checker's overlap and carry
    proxies leave the mechanism alone. The simulator never uses this."""

    BOX = ((L.BACK_X0 + L.BACK_T / 2, 0.0, L.TOP + L.BACK_H / 2), (L.BACK_T, 2 * L.BACK_HALF_Y, L.BACK_H))

    def offline_box(self):
        return tuple(float(v) for v in self.BOX[0]), tuple(float(v) for v in self.BOX[1]), 0.0

    def offline_state(self, state):
        return self.rest_state()


CABINET_SPEC = Cabinet(name=CABINET, pos=(0.0, 0.0, L.TOP), color=L.WOOD, mass=None, yaw=0.0,
                       shapes=tuple(frame_shapes()),
                       joints=(_lever(COLD, L.COLD_Y, L.BLUE), _lever(HOT, L.HOT_Y, L.RED)),
                       friction=L.FRICTION, **mk.TIGHT_FIT)
CABINET_SPEC._check()
OBJECTS = [CABINET_SPEC]


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows no hinge. Offline only (never in the simulator), this model gives the scorer: a lever is
# pushed clockwise by a gripper closed into a block whose pad tips meet the lever's height, on the lever's trailing
# side, and it stays where it is left (the joint friction holds it; the stop plate stops it at 90 degrees).
PAD_FACE = 0.005  # a closed pad block's face ahead of the TCP along the closing direction (GPU, shim task: 5.9 mm,
# less the arm's trail)
PAD_SPAN = 0.038  # the closed pads' height above their tips
AHEAD = 0.2  # rad: a block this far past the lever's centre line is on its leading side and pushes nothing
BAND = 0.0105  # the vinyl grip's half height
CLOSED_MAX = 0.3  # a gripper counts as a closed block under this opening (0 closed .. 1 open)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.q = {COLD: 0.0, HOT: 0.0}

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        for tcp, tips, opening in read:
            if opening >= CLOSED_MAX:
                continue
            tip_z = float(tips[2]) - L.TOP
            for name, y in ((COLD, L.COLD_Y), (HOT, L.HOT_Y)):
                wx, wy = float(tcp[0]) - L.STEM_X, float(tcp[1]) - y
                r = math.hypot(wx, wy)
                if not (0.040 <= r <= L.BAR_L + 0.010):
                    continue
                grip = r >= L.BAR_L - L.GRIP_L
                half = L.GRIP_R if grip else L.BAR_W / 2
                band = BAND if grip else L.BAR_T / 2
                if not (tip_z <= L.LEVER_Z + band and tip_z + PAD_SPAN >= L.LEVER_Z - band):
                    continue
                phi = _wrap(math.pi - math.atan2(wy, wx))  # clockwise from the open direction (-x)
                q = self.q[name]
                if phi > q + AHEAD:
                    continue
                self.q[name] = min(L.Q_CLOSED, max(q, phi + (PAD_FACE + half) / r))

    def states(self) -> dict:
        s = mk.state_from_poses(CABINET_SPEC, (0.0, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0), joints=dict(self.q),
                                source="offline")
        return {CABINET: s}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(CABINET), "source", "sim") != "sim"


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
    """With TASK_TRACE=<file>, append one JSON line per control step: the levers and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The cold lever at 88 degrees or more against its stop, the hot lever untouched, both grippers 30 mm clear and
    the lever still for 1 s; held 1 s. This runs first every control step, so it also feeds the tracker (and, without
    a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {CABINET: objects.get(CABINET)}
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
                       left_tips=arms["left"].tips, right_tips=arms["right"].tips,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(CABINET), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the hot lever turned more than 10 degrees or the cabinet moved 5 mm (latched), or unless both grippers are
    30 mm clear of the cold lever and valve and it has been still for 1 s; then 1.0 the cold lever at 88 degrees or more
    (against its stop), 0.5 from 45 up to 88, 0.25 from 15 up to 45, 0 under 15. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. The left arm only: its fingers closed into a block (yaw turning with the lever so the block's closing
# faces stay square to the push), the block set beside the cold lever's grip on its trailing side, 8 mm short of it,
# pad tips 2 mm under the lever's centre line; moved 16 mm into it (the block face 8 mm inside the grip: the arm
# trails about 2 mm per newton, so that is about 3.5 N against the 2.0 N the valve needs), then swept round the stem
# at the grip's middle radius to 4 degrees past the stop (the stop takes the rest); backed off, up, aside.
PAD_TIP = 0.013
CLOSED = 0.0
OPEN = -0.04695
TIP_Z = L.LEVER_Z - 0.002  # pad tips 2 mm under the lever's centre line
TCP_Z = TIP_Z + PAD_TIP + 0.0005  # +0.5 mm: the loaded arm sits a little low
HOVER = 0.190
GAP = 0.008  # the block's face this far short of the grip at the start
PEN = 0.008  # and this far into it while pushing
OVER = math.radians(4.0)  # the sweep ends this far past the stop
Q_END = L.Q_CLOSED + OVER
BACK_OFF = 0.033
PARK = (0.14, 0.31, 0.170)
SWEEP_STEPS = 7
SWEEP_S = 3.5
PUSH_OFF = L.GRIP_R + PAD_FACE  # centre-line of the grip to the TCP, along the push direction, at zero penetration


def block_xy(q: float, pen: float):
    """Where the TCP goes with the lever commanded at ``q`` (rad): the grip's middle radius, the block ``pen`` m into
    the grip (negative: short of it) on the trailing side."""
    ux, uy = L.lever_dir(q)
    vx, vy = L.push_dir(q)
    off = PUSH_OFF - pen
    return (L.STEM_X + L.GRIP_C * ux - off * vx, L.COLD_Y + L.GRIP_C * uy - off * vy)


def yaw_at(q: float) -> float:
    """The gripper yaw that keeps the closed block's closing faces square to the push (yaw 0 closes along y)."""
    return -q


def _w(duration, x, y, z, finger, label, yaw):
    return Waypoint(float(duration), (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def left_plan(q_end: float = Q_END, release: bool = True) -> list:
    x0, y0 = block_xy(0.0, -GAP)
    xa, ya = block_xy(0.0, PEN)
    plan = [
        _w(2.0, x0, y0, HOVER, CLOSED, "over the cold valve, beside the grip on its trailing side, fingers closing",
           yaw_at(0.0)),
        _w(1.5, x0, y0, TCP_Z + 0.030, CLOSED, "down to just above the lever's height", yaw_at(0.0)),
        _w(1.5, x0, y0, TCP_Z, CLOSED, "slowly down: pad tips 2 mm under the lever's centre line", yaw_at(0.0)),
        _w(1.0, xa, ya, TCP_Z, CLOSED, "move 16 mm in: the block face 8 mm into the grip", yaw_at(0.0)),
    ]
    for i in range(1, SWEEP_STEPS + 1):
        q = q_end * i / SWEEP_STEPS
        x, y = block_xy(q, PEN)
        plan.append(_w(SWEEP_S / SWEEP_STEPS, x, y, TCP_Z, CLOSED, "swing the lever clockwise round the stem",
                       yaw_at(q)))
    plan.append(_w(0.3, *block_xy(q_end, PEN), TCP_Z, CLOSED, "settle against the stop", yaw_at(q_end)))
    if not release:
        return plan + [_w(6.0, *block_xy(q_end, PEN), TCP_Z, CLOSED, "keep pressing on the lever", yaw_at(q_end))]
    xb, yb = block_xy(q_end, -BACK_OFF)
    plan += [
        _w(0.8, xb, yb, TCP_Z, CLOSED, "back off 33 mm", yaw_at(q_end)),
        _w(1.0, xb, yb, HOVER, CLOSED, "straight up", yaw_at(q_end)),
        _w(1.5, PARK[0], PARK[1], PARK[2], CLOSED, "aside", yaw_at(q_end)),
    ]
    return plan


# Wrong ways (counterfactuals): where the sweep is commanded to end (rad), or no let-go.
VARIANT_Q = {"short": math.radians(25.0), "half": math.radians(55.0)}
VARIANTS = tuple(VARIANT_Q) + ("hold_on",)


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "hold_on":
        return {"left": left_plan(release=False)}
    return {"left": left_plan(VARIANT_Q.get(variant, Q_END))}


def demo():
    """The left arm. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short    the sweep commanded to 25 degrees: the lever is left near 30 degrees (0.25)
      half     commanded to 55 degrees: the lever is left near 60 degrees (0.5)
      hold_on  the whole sweep and no let-go: the lever on its stop, the block still pressing on it (0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
