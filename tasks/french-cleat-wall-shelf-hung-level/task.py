"""On the station table, real size: a workbench's back upstand (a 30 mm backboard, front face at x 400 mm, 700 mm
wide, 210 mm tall, on two braces) with the bench's end post (a 30 x 30 mm upright, 210 mm tall) against it on the
left, 270 mm from the midline. A French cleat rail is screwed to the backboard (three screws): birch ply, 340 mm
long, its top edge cut at 45 degrees so that it is 11 mm lower at the board than at its front face. A small tool shelf
(birch ply, 320 x 100 x 80 mm, open top and front, 18 mm end panels, 0.53 kg) stands on the table at x 170 to 270 mm
with the matching cleat screwed to its back (three screws, heads inside). Hung, the unit cleat's bevel lies on the
rail's bevel and its back face on the backboard, the shelf level 40 mm over the table.

Goal: the shelf hung on the cleat, its whole cleat engaged, level, its back flat on the wall, hands off. The shelf has
to be lifted at least 12 mm clear of the wall cleat's front edge to go on, carried back until its cleat touches the
wall, and lowered: the bevels then draw it in and it seats. Lowered short of the wall it comes down in front of the
cleat; lowered on one end only it hangs crooked.

The shelf and the wall are mechkit compounds (mechkit.py, next to this file; no declared software rule: the shelf is
held by the cleat's two bevels and gravity alone). The arms are read through rigstate.py (the let-go check).

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
from scoring import SHELF  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Hang the shelf on the cleat rail so it sits flat against the backboard and level, and let go."

_NORMAL = np.array([1.0, 1.0]) / L.S2  # (x, z) unit normal of the bevel, up and toward the wall


# -- the parts --------------------------------------------------------------------------------------------------------
def _slab(p1, p2, side):
    """A turned slab of thickness SLAB_T whose face lies on the bevel line from p1 to p2 (each (x, z)); ``side`` +1
    puts the material above the line (the unit cleat), -1 below it (the wall cleat). Returns a mechkit box (centre
    given in the frame the points are in, y 0)."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    c = (p1 + p2) / 2 + side * _NORMAL * L.SLAB_T / 2
    length = float(np.linalg.norm(p2 - p1))
    return c, length


def shelf_shapes():
    """The shelf in its unit frame (origin at the middle of its underside, x toward the wall)."""
    ply = L.PLY
    out = []
    for s in (-1, 1):  # end panels, 18 mm: what the fingers pinch
        out.append(mk.box((0.0, s * L.END_MID, L.SH_H / 2), (L.SH_D, L.END_T, L.SH_H), ply))
    out.append(mk.box((0.0, 0.0, L.BOT_T / 2), (L.SH_D, 2 * L.INNER, L.BOT_T), ply))
    out.append(mk.box((L.SH_D / 2 - L.BACK_T / 2, 0.0, L.BOT_T + (L.SH_H - L.BOT_T) / 2),
                      (L.BACK_T, 2 * L.INNER, L.SH_H - L.BOT_T), ply))
    for y in L.USCREW_Y:  # three pan-head screws through the back panel into the unit cleat, heads inside the shelf
        out.append(mk.cylinder((L.SH_D / 2 - L.BACK_T - L.SCREW_H / 2, y, L.USCREW_Z), L.SCREW_R, L.SCREW_H, "X",
                               L.SCREW_GREY))
    # the unit cleat: a turned slab on the bevel (lower edge), stairs 1 mm wide above it up to the cleat's top edge
    x0 = L.UC_FRONT_LOCAL
    x1 = L.U_FACE_X1 - L.SH_SEAT_X
    c, length = _slab((x0, L.u_bevel_z(x0)), (x1, L.u_bevel_z(x1)), +1)
    out.append(mk.box((float(c[0]), 0.0, float(c[1])), (length, L.UCLEAT_L, L.SLAB_T), L.CLEAT_PLY,
                      ypr=(0.0, L.PITCH_RAD, 0.0)))
    n = int(round(L.CLEAT_T / L.STEP))
    for i in range(n):
        xa = L.UC_FRONT_LOCAL + i * L.STEP
        bottom = L.u_bevel_z(xa) + 0.0003
        out.append(mk.box((xa + L.STEP / 2, 0.0, (bottom + L.UC_TOP) / 2), (L.STEP, L.UCLEAT_L, L.UC_TOP - bottom),
                          L.CLEAT_PLY))
    return out


def wall_shapes():
    """The wall in its frame (origin at the wall's front face, on the table, y 0): x toward the room is negative."""
    ox = L.WALL_X
    return [mk.box((L.WALL_T / 2, 0.0, L.WALL_H / 2), (L.WALL_T, L.WALL_W, L.WALL_H), L.WALL_COLOR)]


def cleat_shapes():
    """The wall cleat and its screws, in the wall's frame (origin at the wall's front face, on the table)."""
    ox = L.WALL_X
    out = []
    # the wall cleat: a slab below the bevel line, stairs 1 mm wide under it down to the cleat's bottom edge
    xa, xb = L.W_FACE_X0, L.W_FACE_X1
    c, length = _slab((xa, L.bevel_z(xa)), (xb, L.bevel_z(xb)), -1)
    out.append(mk.box((float(c[0]) - ox, 0.0, float(c[1])), (length, L.WCLEAT_L, L.SLAB_T), L.CLEAT_PLY,
                      ypr=(0.0, L.PITCH_RAD, 0.0)))
    x_front = ox - L.WCLEAT_T
    n = int(round(L.WCLEAT_T / L.STEP))
    for i in range(n):
        xl = x_front + i * L.STEP
        xr = xl + L.STEP
        top = L.bevel_z(xr) - 2 * L.SLAB_T / L.S2 + 0.0010  # 1 mm into the slab, below its top face
        out.append(mk.box(((xl + xr) / 2 - ox, 0.0, (L.WCLEAT_Z0 + top) / 2), (L.STEP, L.WCLEAT_L, top - L.WCLEAT_Z0),
                          L.CLEAT_PLY))
    for y in L.SCREW_Y:  # three pan-head screws through the cleat into the wall (heads 1.5 mm proud, below the shelf)
        out.append(mk.cylinder((x_front - L.SCREW_H / 2 - ox, y, L.SCREW_Z), L.SCREW_R, L.SCREW_H, "X", L.SCREW_GREY))
    return out


def batten_shapes():
    """Two braces behind the board, screwed to the bench top (frame: origin at the board's back face, on the table)."""
    x0, x1 = L.WALL_X + L.WALL_T, L.BATTEN_X1
    return [mk.box(((x1 - x0) / 2, s * L.BATTEN_Y, L.BATTEN_T / 2), (x1 - x0, L.BATTEN_W, L.BATTEN_T), L.BATTEN_COLOR)
            for s in (-1, 1)]


WALL = mk.assembly("wall", origin=(L.WALL_X, 0.0, L.TOP), shapes=wall_shapes(), color=L.WALL_COLOR,
                   friction=L.FRICTION, tight_fit=True)
CLEAT = mk.assembly("wall_cleat", origin=(L.WALL_X, 0.0, L.TOP), shapes=cleat_shapes(), color=L.CLEAT_PLY,
                    friction=L.FRICTION, tight_fit=True)
SHELF_PART = mk.assembly(SHELF, origin=(L.SH_START_X, 0.0, L.TOP), mass=L.SH_MASS, shapes=shelf_shapes(),
                         color=L.PLY, friction=L.FRICTION, tight_fit=True)
BATTENS = mk.assembly("battens", origin=(L.WALL_X + L.WALL_T, 0.0, L.TOP), shapes=batten_shapes(),
                      color=L.BATTEN_COLOR, friction=L.FRICTION)
CASING = mk.assembly("casing", origin=((L.SIDE_X0 + L.WALL_X) / 2, (L.SIDE_Y0 + L.SIDE_Y1) / 2, L.TOP),
                     shapes=[mk.box((0.0, 0.0, L.SIDE_H / 2), (L.WALL_X - L.SIDE_X0, L.SIDE_Y1 - L.SIDE_Y0, L.SIDE_H),
                                    L.SIDE_COLOR)], color=L.SIDE_COLOR, friction=L.FRICTION)
OBJECTS = [WALL, CLEAT, BATTENS, CASING, SHELF_PART]
SPECS = {SHELF: SHELF_PART}
COM_X = float(SHELF_PART.total_center_of_mass()[0])  # local x of the centre of mass: where the fingers pinch


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped box rigidly and drops it straight down; it knows neither the cleat nor the wall.
# Offline only (never in the simulator), this model gives the scorer: the shelf carried by the hands that pinch its end
# panels (both: it follows their mean, tilting with their height difference; one: the far end stays on the table), and,
# when the hands let go, the cleat: both ends of the unit cleat's overlap within the bevels' reach of the wall (7.4 mm
# short of the wall at most) seat it (the bevels draw it in, level), one end only leaves it hanging crooked, none drops
# it to the table.
GRIP_DX = L.GRIP_LOCAL_X
TIPS_DOWN = 0.020  # pad tips 20 mm below the end panels' top edge
PAD_TIP = 0.013
HOLD_OPENING = 0.35
CROOKED_DEG = 6.0
FALL_TILT_DEG = 4.0


def _reading(o, psi, phi) -> S.Reading:
    return S.Reading(np.asarray(o, dtype=float), mk.quat_to_matrix(mk.quat_from_ypr(psi, 0.0, phi)))


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.o = np.array([L.SH_START_X, 0.0, 0.0])
        self.psi = 0.0
        self.phi = 0.0
        self.grip: dict = {}  # side -> (TCP minus the unit's origin at the grab)
        self.was_held = False

    def _holding(self, side, tcp, tips, opening) -> bool:
        sgn = 1.0 if side == "left" else -1.0
        rel = float(tips[2]) - L.TOP - float(self.o[2])
        return bool(opening < HOLD_OPENING and abs(float(tcp[1]) - (float(self.o[1]) + sgn * L.END_MID)) <= 0.015
                    and abs(float(tcp[0]) - (float(self.o[0]) + GRIP_DX)) <= 0.030
                    and 0.030 <= rel <= L.SH_H + 0.003)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        held = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            if side in self.grip:  # keeps holding while the fingers stay closed on the panel
                y_panel = float(self.o[1]) + (1.0 if side == "left" else -1.0) * L.END_MID
                if opening < 0.6 and abs(float(tcp[1]) - y_panel) <= 0.03:
                    held[side] = tcp
                continue
            if self._holding(side, tcp, tips, opening):
                self.grip[side] = np.asarray(tcp, dtype=float) - (self.o + np.array([0.0, 0.0, L.TOP]))
                held[side] = tcp
        for side in list(self.grip):
            if side not in held:
                self.grip.pop(side)
        if held:
            imp = {s: np.asarray(t, dtype=float) - self.grip[s] - np.array([0.0, 0.0, L.TOP]) for s, t in held.items()}
            if len(imp) == 2:
                lo, ro = imp["left"], imp["right"]
                self.o = (lo + ro) / 2
                self.phi = math.atan2(float(lo[2] - ro[2]), 2 * L.END_MID)
                self.psi = -float(lo[0] - ro[0]) / (2 * L.END_MID)
            else:
                side, o = next(iter(imp.items()))
                sgn = 1.0 if side == "left" else -1.0
                end = max(float(o[2]), 0.0)
                self.phi = sgn * math.asin(min(1.0, end / (2 * L.END_MID)))
                self.psi = 0.0
                self.o = np.array([o[0], o[1], end / 2])
            self.was_held = True
        elif self.was_held:
            self._settle()
            self.was_held = False

    def _settle(self) -> None:
        """The hands let go: the unit seats, hangs crooked or drops."""
        if abs(math.degrees(self.phi)) > FALL_TILT_DEG or float(self.o[2]) < L.SEAT - 0.012:
            self._drop()
            return
        r = _reading(self.o, self.psi, self.phi)
        lo, hi, engaged = S.cover(r)
        if engaged <= 0.0:
            self._drop()
            return
        ends = []
        for y in (lo, hi):
            lam = (y - float(r.pos[1])) / float(r.rot[1, 1])
            xb = float(S.world(r, (L.UC_BACK_LOCAL, lam, 0.5 * L.UC_TOP))[0])
            ends.append((y, xb, L.WALL_X - L.PATCH <= xb))
        seats = [e for e in ends if e[2]]
        if len(seats) == 2 and engaged >= L.HALF_ENGAGE:
            self.o = np.array([L.SH_SEAT_X, float(self.o[1]), L.SEAT])
            self.psi = self.phi = 0.0
        elif seats:
            y_e = seats[0][0] if len(seats) == 1 else lo
            y_o = hi if y_e == lo else lo
            self.o = np.array([L.SH_SEAT_X, float(self.o[1]), L.SEAT])
            self.psi = 0.0
            phi = math.radians(CROOKED_DEG) * (1.0 if y_o < y_e else -1.0)
            p = np.array([L.WALL_X - 0.006, y_e, L.SEAT + 0.016])
            rx = mk.axis_angle_matrix((1.0, 0.0, 0.0), phi)
            self.o = p + rx @ (self.o - p)
            self.phi = phi
        else:
            self._drop()

    def _drop(self) -> None:
        self.o = np.array([float(self.o[0]), float(self.o[1]), 0.0])
        self.phi = 0.0

    def states(self) -> dict:
        q = mk.quat_from_ypr(self.psi, 0.0, self.phi)
        return {SHELF: mk.state_from_poses(SHELF_PART, (float(self.o[0]), float(self.o[1]), L.TOP + float(self.o[2]),
                                                          *q), source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(SHELF), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, engaged mm, level deg, back_a mm, back_b mm, x mm]


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
    """With TASK_TRACE=<file>, append one JSON line per control step: the shelf's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The shelf hung on the cleat at 1.0 (both ends in bevel contact, the whole unit cleat engaged, seated within
    1.5 mm of the wall, level and plumb within 1 degree), both grippers 30 mm clear and the shelf still for 1 s; held
    1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline
    stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {SHELF: objects.get(SHELF)}
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
    row["source"] = getattr(parts.get(SHELF), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["engaged_mm"], d["level_deg"], d["back_a_mm"], d["back_b_mm"],
                      d["shelf_mm"][0]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless both grippers are 30 mm clear and the shelf still for 1 s, and 0 with no bevel contact. 1.0 hung with
    both ends in bevel contact, the whole unit cleat engaged, seated within 1.5 mm of the wall, 40 mm up, level and
    plumb within 1 degree; 0.5 both ends in contact, at least half the cleat engaged, within 8 mm of the wall,
    within 3 degrees; 0.25 contact at one end only or at both outside those limits. The score of the latest scored
    step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the 18 mm panel).
OPEN = -0.04695  # 94 mm
GRIP = -(L.END_T - 0.008) / 2  # 10 mm commanded across the 18 mm end panel: about 20 N a finger
YAW = 0.0  # fingers close along y

GRIP_TCP = L.SH_H - TIPS_DOWN + PAD_TIP  # TCP height (over the table) with the tips 20 mm below the panels' top
LIFT = 0.075  # commanded: the arms trail about 5 mm under their 2.6 N share, so the real lift is about 70 mm and the
# unit cleat's lowest point (10 mm over the underside) about 80 mm up: 21 mm over the wall cleat's front edge (59 mm)
PRESS_X = 0.004  # carried 4 mm past where its back meets the wall: pressed back on the wall
PRESS_Z = 0.004  # lowered 4 mm past the seat: it rests its weight on the cleat before the fingers open
HOVER = 0.150
BACK_X = 0.170  # the hands back off to here (118 mm in front of the seated shelf's front face)
SEAT_TCP = GRIP_TCP + L.SEAT


def grip_x(extra=0.0):
    return L.SH_START_X + GRIP_DX + extra


def end_x(extra=0.0):
    return L.SH_SEAT_X + GRIP_DX + extra


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, YAW)


def arm_moves(side, dx=0.0, dy=0.0):
    """One arm: down onto its end panel, pinch it, lift, carry the shelf back to the wall, lower, let go, up, back off.
    ``dx`` (m): this hand's carry ends that far short of (negative) or past the wall; ``dy``: both hands carry the
    shelf that far along the wall (counterfactuals only)."""
    sgn = 1.0 if side == "left" else -1.0
    y0 = sgn * L.END_MID
    y1 = y0 + dy
    x0, x1 = grip_x(), end_x(PRESS_X + dx)
    return [
        _w(2.0, x0, y0, HOVER, OPEN, "over the end panel"),
        _w(1.0, x0, y0, GRIP_TCP + 0.025, OPEN, "down to just above it"),
        _w(1.5, x0, y0, GRIP_TCP, OPEN, "slowly down round it"),
        _w(0.5, x0, y0, GRIP_TCP, OPEN, "settle"),
        _w(1.5, x0, y0, GRIP_TCP, GRIP, "pinch the end panel"),
        _w(2.0, x0, y0, GRIP_TCP + LIFT, GRIP, "lift the shelf clear of the cleat"),
        _w(2.5, x1, y1, GRIP_TCP + LIFT, GRIP, "carry it back until its cleat meets the wall"),
        _w(2.5, x1, y1, SEAT_TCP - PRESS_Z, GRIP, "lower it onto the cleat"),
        _w(0.5, x1, y1, SEAT_TCP - PRESS_Z, GRIP, "settle"),
        _w(1.5, x1, y1, SEAT_TCP - PRESS_Z, OPEN, "let go"),
        _w(1.2, x1, y1, HOVER, OPEN, "straight up"),
        _w(1.5, BACK_X, y1, HOVER, OPEN, "back off"),
    ]


# Wrong ways (counterfactuals): (left dx, right dx, dy).
VARIANT = {"short": (-0.014, -0.014, 0.0), "skew": (0.0, -0.012, 0.0), "shifted": (0.0, 0.0, 0.060)}
VARIANTS = tuple(VARIANT)


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant (the same clock for both)."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    dl, dr, dy = VARIANT.get(variant, (0.0, 0.0, 0.0))
    return {"left": arm_moves("left", dl, dy), "right": arm_moves("right", dr, dy)}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short    both hands stop 14 mm short of the wall, so the shelf is lowered in front of the cleat (0.0)
      skew     the right hand stops 12 mm short, so the shelf is lowered skewed and hangs on one end (0.25)
      shifted  the shelf carried 60 mm along the wall: its cleat overhangs the wall cleat's end (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
