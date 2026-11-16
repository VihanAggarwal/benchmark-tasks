"""On the station table, real size: a 30-pocket moulded egg flat (left) with six USDA large eggs standing in it pointed
end down, and in the middle a half-dozen (2 x 3) moulded-pulp egg carton in its pack-out cradle with its lid hinged on
the side toward the arms and standing open at 110 degrees against its hinge stop.

Goal: stand each of the six eggs pointed end down in a carton cell, close the lid, and let go. An egg dropped or thrown
cracks; an egg standing high keeps the lid from closing.

The flat, the carton (a hinge link: its lid) and the eggs (a stack of twelve discs turned on the egg's axis) are mechkit
assemblies (mechkit.py next to this file). The eggs and the pulp are rigid: a crack is a declared rule read from the
egg's speed (0.8 m/s or more, a fall of about 33 mm or a throw) in scoring.py. The arms are read through rigstate.py
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
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Pack all six eggs into the carton, one to a cell, pointed end down, without cracking any, "
               "and close the lid.")


# -- the egg flat and the carton ---------------------------------------------------------------------------------------
def _boxes(slabs, color):
    return [mk.box(c, s, color, ypr=y) for c, s, y in slabs]


def flat_shapes():
    shapes = [mk.box((0.0, 0.0, L.FLOOR_T / 2), (L.F_PLATE[0], L.F_PLATE[1], L.FLOOR_T), L.PULP_COLOR)]
    for i in range(L.F_ROWS):
        for j in range(L.F_COLS):
            px, py = L.flat_pocket(i, j)
            shapes += _boxes(L.funnel_slabs(px - L.FX, py - L.FY, L.F_R, L.F_r, L.F_D, L.FLOOR_T), L.PULP_COLOR)
    return shapes


def carton_shapes():
    shapes = [mk.box((0.0, 0.0, L.FLOOR_T / 2), (L.C_LX, L.C_LY, L.FLOOR_T), L.PULP_COLOR)]
    for dx in L.C_CELL_X:
        for dy in L.C_CELL_Y:
            shapes += _boxes(L.funnel_slabs(dx, dy, L.C_R, L.C_r, L.C_D, L.FLOOR_T), L.PULP_COLOR)
    t, h = L.PULP_T, L.C_RIM
    for s in (-1, 1):  # the outer walls up to the rim
        shapes.append(mk.box((s * (L.C_LX / 2 - t / 2), 0.0, h / 2), (t, L.C_LY, h), L.PULP_COLOR))
        shapes.append(mk.box((0.0, s * (L.C_LY / 2 - t / 2), h / 2), (L.C_LX, t, h), L.PULP_COLOR))
    return shapes


def cradle_shapes():
    """The pack-out cradle: four low rails round the carton's foot (1 mm clear), 6 mm tall, 3 mm wide."""
    gap, w, h = 0.001, 0.003, 0.006
    ox, oy = L.C_LX / 2 + gap + w / 2, L.C_LY / 2 + gap + w / 2
    rail = (0.30, 0.30, 0.32)
    out = []
    for s in (-1, 1):
        out.append(mk.box((s * ox, 0.0, h / 2), (w, L.C_LY + 2 * (gap + w), h), rail))
        out.append(mk.box((0.0, s * oy, h / 2), (L.C_LX + 2 * (gap + w), w, h), rail))
    return out


FLAT = mk.assembly(L.FLAT_NAME, origin=(L.FX, L.FY, L.TOP), shapes=flat_shapes(), color=L.PULP_COLOR,
                   friction=L.PULP_FRICTION, tight_fit=True)

LID = mk.hinge("lid", "Y", (L.HINGE_X, 0.0, L.C_RIM), _boxes(L.lid_boxes(), L.PULP_COLOR), upper=L.LID_OPEN,
               lower=0.0, mass=L.LID_MASS, damping=L.LID_DAMP)
CARTON = mk.assembly(L.CARTON_NAME, origin=(L.CX, L.CY, L.TOP), shapes=carton_shapes() + cradle_shapes(),
                     joints=[LID], color=L.PULP_COLOR, friction=L.PULP_FRICTION, tight_fit=True)


# -- the eggs: twelve discs turned on the egg's axis ------------------------------------------------------------------
def egg(k: int):
    shapes = []
    for r, z, u0 in zip(L.SLICE_R, L.SLICE_Z, L.SLICE_U0):
        color = L.EGG_COLOR if u0 + L.SLICE_H / 2 < L.EGG_L / 2 else L.EGG_TOP_COLOR
        shapes.append(mk.cylinder((0.0, 0.0, z), r, L.SLICE_H, "Z", color))
    x, y = L.EGG_START[k]
    return mk.assembly(f"e{k}", origin=(x, y, L.TOP + L.EGG_START_Z), mass=L.EGG_MASS, shapes=shapes,
                       color=L.EGG_COLOR, friction=L.EGG_FRICTION, tight_fit=True)


EGG_PARTS = [egg(k) for k in range(6)]
OBJECTS = [FLAT, CARTON, *EGG_PARTS]
SPECS = {p.name: p for p in (*EGG_PARTS, CARTON)}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped egg rigidly (as its bounding box) and drops it when the grip opens; it knows
# nothing of pockets, the carton's lid or the egg's wedge. Offline only (never in the simulator), this model carries an
# egg with the closed gripper that took it, sets it down in the pocket (or cell) it is released over, and turns the
# lid as a closed gripper's pad tip pushes its outer face (the lid then falls shut under its weight unless an egg
# stands too high), as the mechanism and the declared rules would.
GRASP_GAP = 0.040  # a gripper closing to this gap over an egg at its grip height takes it
LET_GAP = 0.042  # opening past this lets it go
PAD_FACE = 0.004  # a closed gripper's pad tip leads its TCP by this much, 13 mm below it
TIP_TO_TCP = 0.0575  # the TCP above an egg's pointed end when held at the grip height (pads on the blunt-end discs)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = {n: np.array([x, y, L.TOP + L.EGG_START_Z]) for n, (x, y) in zip(L.EGGS, L.EGG_START)}
        self.prev = {n: p.copy() for n, p in self.pos.items()}
        self.vel = {n: np.zeros(3) for n in L.EGGS}
        self.held: dict = {}  # arm -> (egg, offset)
        self.psi = L.LID_OPEN_DEG  # the lid's angle above closed: 110 open .. 0 closed

    @staticmethod
    def pockets():
        """Every pocket an egg can be set down in: (x, y, centre height of a seated egg)."""
        out = [(x, y, L.TOP + L.EGG_Z_SEATED) for x, y in L.CELLS]
        for i in range(L.F_ROWS):
            for j in range(L.F_COLS):
                x, y = L.flat_pocket(i, j)
                out.append((x, y, L.TOP + L.EGG_START_Z))
        return out

    def _release(self, name: str) -> None:
        p = self.pos[name]
        best = min(self.pockets(), key=lambda c: math.hypot(p[0] - c[0], p[1] - c[1]))
        if math.hypot(p[0] - best[0], p[1] - best[1]) <= 0.015:
            fall = max(0.0, float(p[2]) - best[2])
            self.pos[name] = np.array([best[0], best[1], best[2]])
        else:  # set down on the table anywhere else: standing, not in a cell
            fall = max(0.0, float(p[2]) - (L.TOP + L.EGG_L / 2))
            self.pos[name] = np.array([p[0], p[1], L.TOP + L.EGG_L / 2])
        self.vel[name] = np.array([0.0, 0.0, -math.sqrt(2.0 * L.G * fall)])

    def _lid_floor(self) -> float:
        """The lowest the lid can fall: an egg standing higher than the lid's inside height keeps it open."""
        for n in L.EGGS:
            p = self.pos[n]
            if S.cell_of(p - np.array([0.0, 0.0, L.TOP])) is None:
                continue
            if float(p[2]) - L.TOP + L.EGG_L / 2 > L.C_RIM + L.LID_CLEAR - 0.0005:
                return 15.0
        return 0.0

    def _push(self, tip) -> bool:
        """A closed gripper's pad tip at ``tip`` (world) against the lid's outer face: turn the lid so the face is
        out of it. True while the tip is touching."""
        hx = L.CX + L.HINGE_X
        if abs(float(tip[1]) - L.CY) > L.LID_Y_HALF:
            return False
        u0, v0 = float(tip[0]) - hx, float(tip[2]) - L.TOP - L.C_RIM
        face = L.LID_CLEAR + L.LID_T

        def coords(psi_deg):
            p = math.radians(psi_deg)
            return (u0 * math.cos(p) + v0 * math.sin(p), -u0 * math.sin(p) + v0 * math.cos(p))  # (along, normal)

        a, d = coords(self.psi)
        if not (0.0 <= a <= L.LID_LEN and 0.0 <= d <= face + 0.0015):
            return False
        if d >= face - 1e-9:
            return True  # touching, nothing to push
        psi = self.psi
        while psi > 0.0:
            psi = max(0.0, psi - 0.25)
            a2, d2 = coords(psi)
            if d2 >= face or not (0.0 <= a2 <= L.LID_LEN):
                break
        self.psi = psi
        return True

    def step(self, objects) -> None:
        arms = rigstate.arms()
        prev_pos = {n: p.copy() for n, p in self.pos.items()}
        self.vel = {n: np.zeros(3) for n in L.EGGS}
        touching = False
        if arms is not None:
            for arm in ("left", "right"):
                a = arms.get(arm) if hasattr(arms, "get") else None
                if a is None:
                    continue
                tcp = np.asarray(a.tcp, dtype=float)
                gap = float(a.opening) * rigstate.FULL_GAP
                if arm in self.held:
                    name, off = self.held[arm]
                    if gap > LET_GAP:
                        del self.held[arm]
                        self._release(name)
                    else:
                        self.pos[name] = tcp + off
                elif gap <= GRASP_GAP:
                    taken = {n for n, _ in self.held.values()}
                    for n in L.EGGS:
                        p = self.pos[n]
                        grip_z = float(p[2]) - L.EGG_L / 2 + TIP_TO_TCP
                        if (n not in taken and math.hypot(p[0] - tcp[0], p[1] - tcp[1]) <= 0.012
                                and abs(float(tcp[2]) - grip_z) <= 0.012):
                            self.held[arm] = (n, p - tcp)
                            break
                    else:
                        if gap < 0.015:
                            touching = self._push(tcp + np.array([PAD_FACE, 0.0, -0.013])) or touching
                else:
                    pass
        if self.psi > 90.0 and not touching:
            self.psi = float(L.LID_OPEN_DEG)  # nothing holds it: it swings back against its hinge stop
        elif self.psi <= 90.0 and not touching:
            self.psi = self._lid_floor()  # shut under its weight, or sits on a standing-high egg
        for n in L.EGGS:
            if self.vel[n][2] == 0.0:
                self.vel[n] = (self.pos[n] - prev_pos[n]) * L.CONTROL_HZ

    def states(self, objects) -> dict:
        out = {}
        for k, spec in enumerate(EGG_PARTS):
            v = self.vel[spec.name]
            out[spec.name] = mk.state_from_poses(spec, (*self.pos[spec.name], *mk.yaw_quat(0.0)),
                                                 (*v, 0.0, 0.0, 0.0), source="offline")
        q = math.radians(L.LID_OPEN_DEG - self.psi)
        out[L.CARTON_NAME] = mk.state_from_poses(CARTON, (*CARTON.pos, *mk.yaw_quat(0.0)), joints={"lid": q},
                                                 source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(L.EGGS[0]), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, seated, lid deg]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: every egg's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Six eggs seated pointed end down, lid closed, no crack, both grippers 30 mm clear and every part still for
    1 s; held 1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the
    offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _fresh()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(parts)
        parts = _OFFLINE.states(parts)
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
    row["source"] = getattr(parts.get("e0"), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["n_seated"], d["lid_deg"], d["egg_z_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if an egg was dropped or thrown, or unless every part is let go and still, or with no egg seated; 1.0 for six
    eggs seated pointed end down with the lid closed; 0.75 with the lid not closed; 0.5 for four or five seated;
    0.25 for one to three. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips
# are 13 mm below. Finger targets: the joint value is minus half the commanded gap.
OPEN = -0.04695  # 94 mm
GRIP = -0.0165  # 33 mm commanded across the 38 mm blunt-end disc (5 mm narrower: about 12 N a finger)
LETGO = -0.0225  # 45 mm: clear of the 38 mm disc and of a seated neighbour's 42 mm
SHUT = -0.001  # closed fingers for pushing the lid
YAW = 0.0  # the left arm closes along y, across the egg
HOVER = 0.135  # carry height: a carried egg's pointed end 77 mm up, over a seated egg's 62 mm top
PICK_TCP = L.EGG_START_Z - L.EGG_L / 2 + TIP_TO_TCP  # 61.5 mm
PLACE_TCP = L.EGG_Z_SEATED - L.EGG_L / 2 + TIP_TO_TCP  # 63.0 mm
PARK = {"left": (0.14, 0.40, 0.150), "right": (0.14, -0.40, 0.150)}
LID_AIM = 0.090  # the pad tip pushes the lid's outer face this far along it from the hinge
LID_ARC_DEG = (110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0)  # the lid angles the right arm's pad follows
LID_INTO = 0.003  # the tip is commanded this far into the face


def _w(d, x, y, z, finger, label, yaw=YAW):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def pick(k: int):
    x, y = L.EGG_START[k]
    return [
        _w(1.5, x, y, HOVER, OPEN, f"over egg {k} in the flat"),
        _w(1.2, x, y, PICK_TCP + 0.020, OPEN, "down to just above the egg"),
        _w(1.0, x, y, PICK_TCP, OPEN, "down beside the egg's blunt end"),
        _w(0.5, x, y, PICK_TCP, OPEN, "settle"),
        _w(1.5, x, y, PICK_TCP, GRIP, "close on the egg"),
        _w(0.5, x, y, PICK_TCP, GRIP, "settle"),
        _w(1.5, x, y, HOVER, GRIP, "lift the egg out of the pocket"),
    ]


def place(cell: int, drop: float = 0.0):
    x, y = L.CELLS[cell]
    return [
        _w(2.0, x, y, HOVER, GRIP, f"carry over cell {cell}"),
        _w(1.5, x, y, PLACE_TCP + 0.020 + drop, GRIP, "down toward the cell"),
        _w(1.2, x, y, PLACE_TCP + drop, GRIP, "lower the egg into the cell"),
        _w(0.5, x, y, PLACE_TCP + drop, GRIP, "settle"),
        _w(1.5, x, y, PLACE_TCP + drop, LETGO, "open the grip a little"),
        _w(1.2, x, y, HOVER, LETGO, "straight up"),
    ]


def lid_tip(psi_deg: float, v: float):
    """World (x, z above the table) of the lid's outer face at LID_AIM along it, v up from its face plane at lid
    angle psi (the pad tip's place), and the TCP that puts the pad tip there."""
    p = math.radians(psi_deg)
    x = L.CX + L.HINGE_X + LID_AIM * math.cos(p) - v * math.sin(p)
    z = L.C_RIM + LID_AIM * math.sin(p) + v * math.cos(p)
    return x, z


def lid_plan(wait: float):
    """The right arm's: parked while the left packs, then its pad (fingers closed, closing along x) pushes the lid's
    outer face along its arc to past vertical, backs off and waits for the lid to shut under its weight."""
    face = L.LID_CLEAR + L.LID_T
    y = L.CY
    ry = math.pi / 2  # fingers closing along x: a wide pad face against the lid
    park = PARK["right"]
    plan = [_w(3.0, *park[:2], park[2], OPEN, "right arm out of the way"),
            _w(wait, *park[:2], park[2], OPEN, "wait while the left arm packs")]
    x0, z0 = lid_tip(LID_ARC_DEG[0], face + 0.025)
    plan.append(_w(3.0, x0 - PAD_FACE, y, z0 + 0.013 + 0.02, SHUT, "over the open lid's back", ry))
    plan.append(_w(2.0, x0 - PAD_FACE, y, z0 + 0.013, SHUT, "down behind the lid's outer face", ry))
    for psi in LID_ARC_DEG:
        x, z = lid_tip(psi, face - LID_INTO)
        plan.append(_w(1.5, x - PAD_FACE, y, z + 0.013, SHUT, f"push the lid to {psi:.0f} degrees", ry))
    xe, ze = lid_tip(LID_ARC_DEG[-1], face - LID_INTO)
    plan.append(_w(1.5, xe - PAD_FACE - 0.040, y, ze + 0.013, SHUT, "back off from the lid", ry))
    plan.append(_w(2.0, *park[:2], park[2], OPEN, "right arm out of the way"))
    plan.append(_w(9.0, *park[:2], park[2], OPEN, "wait while the lid shuts"))
    return plan


# Counterfactuals: where each goes wrong. lidshort: the lid is pushed only to 100 degrees (it swings back open: 0.75);
# drop: the first egg is let go 70 mm over its cell (it hits at 1.2 m/s: it cracks, 0); five: only five eggs are packed
# (0.5).
VARIANTS = ("lidshort", "drop", "five")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the left arm moves the eggs one at a time from the flat into the carton, near row first on the
    right-hand side, then parks. The right arm waits out of the way, then closes the lid."""
    global LID_ARC_DEG
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    left = []
    n_eggs = 5 if variant == "five" else 6
    for k in range(n_eggs):
        left += pick(k) + place(L.CELL_PICK_ORDER[k], drop=0.070 if (variant == "drop" and k == 0) else 0.0)
    left.append(_w(2.5, *PARK["left"][:2], PARK["left"][2], OPEN, "left arm out of the way"))
    arc = LID_ARC_DEG
    if variant == "lidshort":
        LID_ARC_DEG = (110.0, 105.0, 100.0)
    try:
        right = lid_plan(max(0.0, _duration(left) - 3.0))
    finally:
        LID_ARC_DEG = arc
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      lidshort  the lid is pushed to 100 degrees only: it swings back open (0.75)
      drop      the first egg is let go 70 mm over its cell: it hits at 1.2 m/s and cracks (0)
      five      only five eggs are packed (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
