"""On the station table (the desk top), real size: a desk phone, a desktop monitor behind it and a pen pot beside it. The
phone is a loose 1.10 kg charcoal base, 200 mm (x) by 240 mm (y) by 55 mm high, with two rectangular cups sunk 10 mm in
its top (earpiece cup 58 mm long, mouthpiece cup 82 mm long, both with 8 mm to spare each side across and 6 mm each end round the pads that go in them)
and a spring hook-switch plunger standing 8 mm proud of the flat top between them. A loose 0.17 kg charcoal handset
(197 mm long: a 46 mm earpiece pad and a 70 mm mouthpiece pad, both 28 mm tall, joined by an arched round handle 34 mm across
whose underside is 14 mm over the pads' bottoms) lies on the desk to the phone's right, face down on its
two pads, long axis along x, earpiece end pointing away from the robot.

Goal: pick the handset up by its handle, turn it a quarter turn, lower both pads into their own cups (the mouthpiece
pad is too long for the earpiece cup, so it only seats one way round), where its handle presses the hook switch down
4 mm (the call ends), let go and take the hand away, without shoving the phone.

The phone, the handset, the monitor and the pen pot are mechkit compounds (mechkit.py, next to this file); the phone
carries one slider (the hook switch). No software rules: every rule is the scorer's reading of poses and of the hook's
travel. The arms are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Trace: TASK_TRACE=<file> (one JSON line per
control step).
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import HANDSET, PHONE  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "The call is over. Hang up: put the handset back on the phone so each end sits in its own cup of the cradle and the "
    "hook switch is pressed down, then let go. Don't shove the phone about."
)


# -- the parts --------------------------------------------------------------------------------------------------------
def phone_shapes():
    """The phone, base frame at the centre of its underside: a slab to the cups' floor and a top plate 10 mm thick with the
    two cups cut out of it."""
    sx, sy = L.PHONE_SIZE
    holes = [(cx, cy, hx, hy) for (cx, cy), (hx, hy) in (L.EAR_CUP, L.MOUTH_CUP)]
    return [mk.box((0.0, 0.0, L.FLOOR_Z / 2), (sx, sy, L.FLOOR_Z), L.PHONE_CHARCOAL)] + list(
        mk.plate_with_holes((0.0, 0.0, L.FLOOR_Z + L.CUP_DEPTH / 2), (sx, sy), holes, L.CUP_DEPTH, L.PHONE_CHARCOAL))


def hook_link():
    """The hook switch: a round plunger on a slider (downward is positive), 8 mm proud of the top, spring up."""
    top = L.HOOK_FREE_TOP
    shaft = mk.cylinder((L.HOOK_XY[0], L.HOOK_XY[1], top - L.HOOK_BODY / 2), L.HOOK_RADIUS, L.HOOK_BODY, "Z", L.HOOK_RED)
    return mk.slider(L.HOOK, "-Z", [shaft], upper=L.HOOK_TRAVEL, mass=L.HOOK_MASS, spring=L.HOOK_SPRING,
                     damping=L.HOOK_DAMPING, friction=0.0)


def handset_shapes():
    """The handset, base frame at the centre of the pad-bottom plane over the cradle's centre (z up)."""
    out = []
    for name in ("ear", "mouth"):
        (cx, cy), (sx, sy), h = L.PADS[name]
        out.append(mk.box((cx, cy, h / 2), (sx, sy, h), L.HANDSET_CHARCOAL))
    # the handle: a round bar 34 mm across, like a real handset's grip, along the handset's long axis
    out.append(mk.cylinder((0.0, (L.HANDLE_Y0 + L.HANDLE_Y1) / 2, (L.HANDLE_Z0 + L.HANDLE_Z1) / 2), L.HANDLE_X / 2,
                           L.HANDLE_Y1 - L.HANDLE_Y0, "Y", L.HANDSET_CHARCOAL))
    return out


def monitor_shapes():
    """The monitor, base frame on the desk at the foot's centre: a flat foot, a neck and the panel."""
    fx, fy, fz = L.MON_FOOT
    nx, ny, nz = L.MON_NECK
    px, py, pz = L.MON_PANEL
    return [
        mk.box((0.0, 0.0, fz / 2), (fx, fy, fz), L.MON_BLACK),
        mk.box((0.010, 0.0, fz + nz / 2), (nx, ny, nz), L.MON_BLACK),
        mk.box((0.0, 0.0, L.MON_PANEL_Z0 + pz / 2), (px, py, pz), L.MON_BLACK),
    ]


PHONE_PART = mk.assembly(PHONE, origin=L.PHONE_ORIGIN, mass=L.PHONE_MASS, shapes=phone_shapes(),
                         joints=[hook_link()], color=L.PHONE_CHARCOAL, friction=L.PHONE_FRICTION, tight_fit=True)
HANDSET_PART = mk.assembly(HANDSET, origin=L.HANDSET_ORIGIN, mass=L.HANDSET_MASS, shapes=handset_shapes(),
                           yaw=L.HANDSET_YAW, color=L.HANDSET_CHARCOAL, friction=L.HANDSET_FRICTION, tight_fit=True)
MONITOR = mk.assembly("monitor", origin=(L.MON_X, 0.0, L.TOP), shapes=monitor_shapes(), color=L.MON_BLACK)
PEN_POT = mk.assembly("pen_pot", origin=(L.POT_XY[0], L.POT_XY[1], L.TOP),
                      shapes=[mk.cylinder((0.0, 0.0, L.POT_H / 2), L.POT_R, L.POT_H, "Z", L.POT_BLUE)],
                      color=L.POT_BLUE)
OBJECTS = [MONITOR, PEN_POT, PHONE_PART, HANDSET_PART]
SPECS = {HANDSET: HANDSET_PART, PHONE: PHONE_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped box rigidly and drops it onto the bounding box below; it knows nothing of the
# cups or the hook switch. Offline only (never in the simulator), this model gives the scorer the handset: a closed
# gripper whose TCP is within 12 mm of the handset's grip point and 22 to 46 mm over its pad bottoms takes it up and
# carries it, turning with the gripper's heading (read from the demo's own plan); opened, the handset drops straight
# down onto what is under it: the cups' floors when both pads fit their cups, the phone's top if any pad is over it,
# else the desk. The hook switch goes down as far as the handle's underside over it pushes it. The phone never moves.
HOLD_OPEN = 0.60  # opening (0 closed .. 1 open) at or below this: closed on the handle (26 mm commanded is 0.28)
LET_OPEN = 0.70  # above this: let go
GRASP_XY = 0.012


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _rot2(yaw: float) -> np.ndarray:
    return np.array(L.yaw_matrix(yaw), dtype=float)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.o = np.array(L.HANDSET_ORIGIN, dtype=float)
        self.yaw = L.HANDSET_YAW
        self.held = None  # (side, grip offset in the handset's frame (x, y), TCP height over its pad bottoms)
        self.k = 0
        self.plans = None
        self.heading: dict = {}

    def _plan_headings(self, t: float) -> dict:
        """Each planned arm's gripper heading (rad) at ``t`` from the demo's own waypoints (offline stub only)."""
        if self.plans is None:
            self.plans = {}
            try:
                from yam_arena import demo as stub_demo, station

                for arm, wps in dict(demo()).items():
                    self.plans[arm] = stub_demo.ArmPlan(list(wps), station.BASES[arm])
            except Exception:  # noqa: BLE001 - without the plans the handset simply never turns
                self.plans = {}
        out = {}
        for arm, plan in self.plans.items():
            rot = plan.target(t)[1]
            out[arm] = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        return out

    def pads_world(self, o, yaw) -> dict:
        """Each pad's four bottom corners (x, y) in the world for a handset frame at ``o`` turned ``yaw``."""
        r = _rot2(yaw)
        return {n: np.array([o[:2] + r @ np.array(c[:2]) for c in L.pad_corners(n)]) for n in L.PADS}

    def support(self, o, yaw) -> float:
        """The height (the handset frame's z, world) a handset let go over (o, yaw) comes to rest at."""
        pads = self.pads_world(o, yaw)
        base = np.array(L.PHONE_ORIGIN[:2])
        both = True
        any_over = False
        for name, corners in pads.items():
            (cx, cy), (sx, sy) = L.CUPS[name]
            rel = corners - base
            both &= bool(np.all(np.abs(rel[:, 0] - cx) <= sx / 2 + L.SLACK) and np.all(np.abs(rel[:, 1] - cy) <= sy / 2 + L.SLACK))
            any_over |= bool(np.any(np.all(np.abs(rel) <= np.array(L.PHONE_SIZE) / 2, axis=1)))
        if both:
            return L.SEATED_Z
        if any_over:
            return L.TOP + L.PHONE_H
        return L.TOP

    def hook(self) -> float:
        """How far the handle's underside, if it is over the plunger, pushes it down (m)."""
        r = _rot2(self.yaw)
        local = r.T @ (np.array(L.PHONE_ORIGIN[:2]) + np.array(L.HOOK_XY) - self.o[:2])
        over = abs(local[0]) <= L.HANDLE_X / 2 and L.HANDLE_Y0 <= local[1] <= L.HANDLE_Y1
        if not over:
            return 0.0
        under = float(self.o[2]) + L.HANDLE_Z0
        return float(min(max(L.TOP + L.HOOK_FREE_TOP - under, 0.0), L.HOOK_TRAVEL))

    def step(self, arms) -> None:
        self.k += 1
        t = self.k / L.CONTROL_HZ
        heads = self._plan_headings(t)
        dh = {a: _wrap(h - self.heading.get(a, h)) for a, h in heads.items()}
        self.heading.update(heads)
        read = rigstate.read_arms(arms)
        if read is None:
            return
        for side, (tcp, _tips, opening) in zip(rigstate.ARMS, read):
            if self.held is None:
                if opening <= HOLD_OPEN:
                    grip_xy = self.o[:2] + _rot2(self.yaw) @ np.array([0.0, L.GRIP_Y])
                    dz = float(tcp[2] - self.o[2])
                    if (math.hypot(tcp[0] - grip_xy[0], tcp[1] - grip_xy[1]) <= GRASP_XY
                            and abs(dz - L.GRIP_Z) <= 0.012):
                        off = _rot2(self.yaw).T @ (np.array(tcp[:2]) - self.o[:2])
                        self.held = (side, off, dz)
            elif self.held[0] == side:
                if opening > LET_OPEN:
                    self.held = None
                    self.o[2] = self.support(self.o, self.yaw)
                else:
                    self.yaw = _wrap(self.yaw + dh.get(side, 0.0))
                    _, off, dz = self.held
                    self.o[:2] = np.array(tcp[:2]) - _rot2(self.yaw) @ off
                    self.o[2] = float(tcp[2]) - dz

    def states(self) -> dict:
        q = (0.0, 0.0, 0.0, 1.0)
        return {
            HANDSET: mk.state_from_poses(HANDSET_PART, (*self.o, *mk.yaw_quat(self.yaw)), source="offline"),
            PHONE: mk.state_from_poses(PHONE_PART, (*L.PHONE_ORIGIN, *q), joints={L.HOOK: self.hook()},
                                       source="offline"),
        }


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(HANDSET), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def _restart() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@rigstate.on_reset
def _new_episode() -> None:
    _restart()


@mk.on_reset
def _kit_reset() -> None:
    _restart()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: both parts' poses and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The handset seated in its cups, the hook switch down, the phone unshoved, both grippers 30 mm clear and both parts
    still for 1 s; held 1 s. This runs first every control step, so it also feeds the tracker (and, without a
    simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _restart()
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
    row["source"] = getattr(parts.get(HANDSET), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a part ever left the desk, the phone was shoved 8 mm or tilted 3 deg,
    or unless both hands are clear and both parts still. Else 1.0 both pads in their own cups, level, square and the hook
    switch down 2 mm or more; 0.5 one or both pads in their own cups and both resting on the phone, not seated; 0.25 both pads resting on
    the phone, neither in its cup; 0 otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
OPEN = -0.04695  # 94 mm
HANDLE_GRIP = -(L.GRIP_W - 0.008) / 2  # 26 mm commanded across the handle's 34 mm
# The right arm does it all. It picks the handset at yaw 0 (the fingers close along y, across the handle, which lies
# along x) and turns it a quarter turn to the left (yaw +pi/2) while carrying it, so the earpiece end swings from +x to
# +y over the cradle: yaw +pi/2 is the right arm's workable heading over the midline (-pi/2 runs joint6 into its limit).
YAW_PICK = 0.0
# Trim of the place yaw, for the GPU run: parts carried in a gripper mostly ended turned counter-clockwise from the
# commanded yaw by about +0.6 to +1.7 degrees in our other tasks' runs, so the handset is laid 1.2 degrees clockwise.
# Offline (no bias) it lies 1.2 degrees off, well inside the 6 degree window. If handset_yaw_vs_phone_deg in TASK_TRACE
# after the release is more than 2 degrees from 0, change YAW_TRIM by minus that many degrees.
YAW_TRIM = math.radians(-1.2)
YAW_PLACE = math.pi / 2 + YAW_TRIM
PRESS = 0.002  # lowered 2 mm past the cups' floor, so the pads rest their full weight there before the fingers open
APPROACH = 0.030  # each final descent starts this far above its end height and takes 1.5 to 2 s
HOVER = L.TOP + 0.12  # pad tips 107 mm over the desk, 59 mm over the handle's top


def _grip_xy(origin_xy, yaw):
    r = np.array(L.yaw_matrix(yaw))
    return tuple(float(v) for v in np.array(origin_xy) + r @ np.array([0.0, L.GRIP_Y]))


PICK_XY = _grip_xy(L.HANDSET_ORIGIN[:2], L.HANDSET_YAW)  # (0.31, -0.22)
PLACE_XY = _grip_xy(L.PHONE_ORIGIN[:2], 0.0)  # (0.30, -0.010)
GRIP_TCP = L.TOP + L.GRIP_Z  # TCP height when picking: pad tips 21 mm up the handle
SET_TCP = L.SEATED_Z + L.GRIP_Z - PRESS
CARRY_TCP = L.SEATED_Z + L.GRIP_Z + APPROACH  # pad bottoms 75 mm over the desk: 20 mm over the phone's top


def _w(d, xy, z, finger, label, yaw):
    return Waypoint(d, (xy[0], xy[1], z), finger, label, yaw)


def right_plan():
    """The right arm: over the handle, down beside it, grip across it, lift to 75 mm, carry over the cradle turning a quarter
    turn, lower both pads into their cups, let go, straight up, away back over where the handset lay."""
    return [
        _w(2.5, PICK_XY, HOVER, OPEN, "over the handset's handle", YAW_PICK),
        _w(1.0, PICK_XY, GRIP_TCP + APPROACH, OPEN, "down to just above the grip", YAW_PICK),
        _w(1.5, PICK_XY, GRIP_TCP, OPEN, "slowly down beside the handle", YAW_PICK),
        _w(0.5, PICK_XY, GRIP_TCP, OPEN, "settle", YAW_PICK),
        _w(1.5, PICK_XY, GRIP_TCP, HANDLE_GRIP, "grip the handle", YAW_PICK),
        _w(0.5, PICK_XY, GRIP_TCP, HANDLE_GRIP, "settle the grip", YAW_PICK),
        _w(1.5, PICK_XY, CARRY_TCP, HANDLE_GRIP, "lift it clear of the desk", YAW_PICK),
        _w(3.0, PLACE_XY, CARRY_TCP, HANDLE_GRIP, "carry it over the cradle, a quarter turn to the left", YAW_PLACE),
        _w(0.5, PLACE_XY, CARRY_TCP, HANDLE_GRIP, "settle", YAW_PLACE),
        _w(2.0, PLACE_XY, SET_TCP, HANDLE_GRIP, "slowly down, each pad into its cup", YAW_PLACE),
        _w(0.5, PLACE_XY, SET_TCP, HANDLE_GRIP, "settle", YAW_PLACE),
        _w(1.5, PLACE_XY, SET_TCP, OPEN, "let go", YAW_PLACE),
        _w(1.5, PLACE_XY, HOVER, OPEN, "straight up", YAW_PLACE),
        _w(2.0, PICK_XY, HOVER, OPEN, "away over where the handset lay", YAW_PLACE),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check). The left arm has no plan: it stays at its
    ready pose with the gripper open, out of the way."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"right": right_plan()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
