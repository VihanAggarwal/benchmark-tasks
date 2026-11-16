"""On the station table, real size: a rigid two-piece setup box. The open base (2 mm greyboard, 70 x 114 x 40 mm,
0.05 kg) stands at x 330 mm, y -70 mm with a pocket laser distance meter in its hard case (55 x 105 x 28 mm, 0.12 kg)
lying flat in it. The matching telescoping lid (2 mm board, 79 x 123 mm outside, 36 mm tall, 0.05 kg) stands top up on
the table at x 330 mm, y -290 mm. Its inside is 2.5 mm larger than the base's outside on every side.

Goal: close the box: carry the lid over the base and lower it square, so it telescopes down over the base's walls until
its top panel rests on the base's rim (top outer face 42 mm over the table), the box left where it was, the product
inside, hands off. A lid brought down yawed, tilted or more than 2.5 mm out of line catches on the base's rim and
rests on it, or wedges half on.

The base, the lid and the product are mechkit compounds (mechkit.py, next to this file; the base and lid are boxes of
2 mm walls, the product one solid). There is no declared software rule: the task is rigid bodies and friction. The arms
are read through rigstate.py (the let-go check).

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
from scoring import BASE, LID, PRODUCT  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Close the box: seat the lid fully over the base with the product inside, square and without wedging it. Leave the box where it is."


# -- the parts --------------------------------------------------------------------------------------------------------
def base_shapes():
    return list(mk.open_box((0.0, 0.0, L.BASE_H / 2), (L.BASE_LX, L.BASE_LY, L.BASE_H), L.WALL, L.FLOOR,
                            L.BASE_COLOR))


def lid_shapes():
    w, s = L.WALL, L.LID_SKIRT
    out = [mk.box((0.0, 0.0, s + L.LID_PANEL / 2), (L.LID_OX, L.LID_OY, L.LID_PANEL), L.LID_COLOR)]
    for sx in (-1, 1):
        out.append(mk.box((sx * (L.LID_OX / 2 - w / 2), 0.0, s / 2), (w, L.LID_OY, s), L.LID_COLOR))
    for sy in (-1, 1):
        out.append(mk.box((0.0, sy * (L.LID_OY / 2 - w / 2), s / 2), (L.LID_OX - 2 * w, w, s), L.LID_COLOR))
    return out


BASE_PART = mk.assembly(BASE, origin=(L.BASE_X, L.BASE_Y, L.TOP), mass=L.BASE_MASS, shapes=base_shapes(),
                        color=L.BASE_COLOR, friction=L.FRICTION, tight_fit=True)
LID_PART = mk.assembly(LID, origin=(L.LID_X, L.LID_Y, L.TOP), mass=L.LID_MASS, shapes=lid_shapes(),
                       color=L.LID_COLOR, friction=L.FRICTION, tight_fit=True)
PRODUCT_PART = mk.assembly(PRODUCT, origin=(L.BASE_X, L.BASE_Y, L.TOP + L.PR_Z), mass=L.PR_MASS,
                           shapes=[mk.box((0.0, 0.0, 0.0), (L.PR_LX, L.PR_LY, L.PR_H), L.PR_COLOR)],
                           color=L.PR_COLOR, friction=L.FRICTION, tight_fit=True)
OBJECTS = [BASE_PART, LID_PART, PRODUCT_PART]
SPECS = {BASE: BASE_PART, LID: LID_PART, PRODUCT: PRODUCT_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and, on release, drops it onto the highest thing's bounding box
# under it; it knows nothing of a lid telescoping over a base. Offline only (never in the simulator), this model
# gives the scorer: the lid carried by a gripper whose pads close on its two long skirt walls, and on release landing
# (1) seated if its centre is within the 2.5 mm play of the base's, (2) on the base's rim if its footprint overlaps the
# base's, (3) on the table otherwise; while carried it cannot go below those heights. The base and the product stay
# where they began.
GRIP_OPENING = 0.85  # a gripper commanded onto the lid reads about 0.76 (the gap 71 of 94 mm); open reads 1.0


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.lid = np.array([L.LID_X, L.LID_Y, 0.0])  # the lid's frame origin (x, y, rim height over the table)
        self.grip: dict = {}  # side -> offset of the TCP from the lid's frame origin

    @staticmethod
    def floor_z(x: float, y: float) -> float:
        """The lowest rim height the lid can have at (x, y): seated, on the base's rim, or the table."""
        dx, dy = x - L.BASE_X, y - L.BASE_Y
        if abs(dx) <= L.CLR + 1e-9 and abs(dy) <= L.CLR + 1e-9:
            return L.SEAT_RIM
        if abs(dx) <= (L.BASE_LX + L.LID_OX) / 2 and abs(dy) <= (L.BASE_LY + L.LID_OY) / 2:
            return L.BASE_H
        return 0.0

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            rel = np.array([float(tcp[0]) - self.lid[0], float(tcp[1]) - self.lid[1],
                            float(tcp[2]) - L.TOP - self.lid[2]])
            holding = (opening < GRIP_OPENING and abs(rel[0]) <= 0.012 and abs(rel[1]) <= L.LID_OY / 2
                       and 0.005 <= rel[2] <= 0.035)
            if side in self.grip:
                if opening >= GRIP_OPENING:
                    self.grip.pop(side)
                    self.lid[2] = self.floor_z(self.lid[0], self.lid[1])  # let go: it drops onto what is under it
                continue
            if holding:
                self.grip[side] = rel
        for side, off in self.grip.items():
            tcp = read[rigstate.ARMS.index(side)][0]
            x, y, z = float(tcp[0]) - off[0], float(tcp[1]) - off[1], float(tcp[2]) - L.TOP - off[2]
            self.lid = np.array([x, y, max(z, self.floor_z(x, y))])

    def states(self) -> dict:
        q = mk.yaw_quat(0.0)
        out = {}
        for part, spec in SPECS.items():
            if part == LID:
                pose = (self.lid[0], self.lid[1], L.TOP + self.lid[2], *q)
            else:
                pose = (*spec.pos, *q)
            out[part] = mk.state_from_poses(spec, pose, source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(LID), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, lid top mm, lid offset mm, yaw gap deg, tilt deg]


def _fresh() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


@rigstate.on_reset
def _new_episode() -> None:
    _fresh()


@mk.on_reset
def _kit_reset() -> None:
    _fresh()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: every key part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The lid seated, square and centred on the base, the base still and in place, the product inside, both grippers
    30 mm clear and every part still for 1 s, no latched fall; held 1 s. This runs first every control step, so it
    also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _fresh()
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
    row["source"] = getattr(parts.get(LID), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["lid_top_mm"], d["lid_offset_mm"], d["lid_yaw_gap_deg"], d["lid_tilt_deg"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the lid or the base fell off the table or the product left the base, or unless every part is let go and
    still, or the base was moved over 20 mm; 1.0 the lid seated (top face 42 mm over the table within 1.5 mm), square
    (yaw within 1.5 degrees, tilt under 1) and centred (3.5 mm); 0.5 the lid engaged 10 mm or more over the base but
    not seated or square; 0.25 the lid resting on the base otherwise. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger target: the joint value is minus half the commanded gap (8 mm narrower than the lid).
OPEN = -0.04695  # 94 mm: 7.5 mm each side of the lid's 79 mm
GRIP = -(L.LID_OX - 0.008) / 2  # 71 mm commanded across the lid
YAW = math.pi / 2  # the right arm closes along x (across the lid's short side)
LID_GRIP_Z = 0.019  # the TCP over the lid's rim: pad tips 6 mm up its 34 mm skirt wall
HOVER = 0.100
CARRY_TCP = L.BASE_H + 0.010 + LID_GRIP_Z  # the rim 10 mm over the base's rim
ENTRY_TCP = L.BASE_H + 0.003 + LID_GRIP_Z  # the rim 3 mm over the base's rim
SEAT_TCP = L.SEAT_RIM + LID_GRIP_Z  # the lid seated: 25 mm
PRESS = 0.001  # lowered 1 mm past that, so the panel rests its weight on the rim before the fingers open
X_TRIM, Y_TRIM = 0.0, 0.0  # calibration of where the lid is put over the base (tune on the GPU)
PARK = (0.25, -0.36, 0.150)


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, YAW)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def pick_up():
    x, y = L.LID_X, L.LID_Y
    return [
        _w(2.5, x, y, HOVER, OPEN, "over the lid"),
        _w(1.0, x, y, LID_GRIP_Z + 0.020, OPEN, "down to just above the rim"),
        _w(1.5, x, y, LID_GRIP_Z, OPEN, "slowly down: pads on the long skirt walls"),
        _w(0.5, x, y, LID_GRIP_Z, OPEN, "settle"),
        _w(1.5, x, y, LID_GRIP_Z, GRIP, "grip the lid across its skirt"),
        _w(2.0, x, y, CARRY_TCP, GRIP, "lift it clear of the base's height"),
    ]


def close_box(dx=0.0, dy=0.0):
    """Carry the lid over the base (``dx``, ``dy`` put it off the base's line: a counterfactual), lower it onto the
    rim, then down over the walls until the panel rests on the rim, let go, up, aside."""
    x, y = L.BASE_X + X_TRIM + dx, L.BASE_Y + Y_TRIM + dy
    return [
        _w(3.5, x, y, CARRY_TCP, GRIP, "over the base"),
        _w(1.0, x, y, CARRY_TCP, GRIP, "settle over the base"),
        _w(1.5, x, y, ENTRY_TCP, GRIP, "down to 3 mm over the base's rim"),
        _w(0.5, x, y, ENTRY_TCP, GRIP, "settle"),
        _w(3.0, x, y, SEAT_TCP - PRESS, GRIP, "lower it over the walls until the panel rests on the rim"),
        _w(0.5, x, y, SEAT_TCP - PRESS, GRIP, "settle"),
        _w(1.5, x, y, SEAT_TCP - PRESS, OPEN, "let go"),
        _w(1.5, x, y, HOVER, OPEN, "straight up"),
        _w(1.5, *PARK[:2], PARK[2], OPEN, "aside"),
    ]


# Wrong ways (counterfactuals): where the lid is put over the base.
VARIANT_OFFSET = {"off_x": (0.005, 0.0), "off_y": (0.0, 0.005)}  # 5 mm out of line (play is 2.5 mm per side)
VARIANTS = tuple(VARIANT_OFFSET)


def plans(variant: str = "") -> dict:
    """The right arm's plan for the demo or a counterfactual variant. The left arm has none (it holds the ready pose).

    Order of work: pick the lid up from the table, carry it over the base, lower it square, let go, move aside."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    dx, dy = VARIANT_OFFSET.get(variant, (0.0, 0.0))
    return {"right": pick_up() + close_box(dx, dy)}


def demo():
    """The right arm alone (the lid, the base and the product are all within its reach); the left holds the ready
    pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      off_x   the lid put over the base 5 mm out of line along x: it catches on the rim (0.25)
      off_y   the same along y (0.25)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
