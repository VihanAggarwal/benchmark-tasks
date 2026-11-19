"""On the station table, real size: a mini centrifuge (Corning LSE 6770 class, 174 x 148 x 122 mm, 1.1 kg) stands
loose at x 150 mm, y 0, its lid shut and latched, the 8-place tube rotor snapped on its shaft. The 4-place strip
rotor lies hub up on the table at x 250, y +200 mm, a quarter turn off its key; two equally filled 8-tube PCR strips
stand in columns 4 and 9 of a 96-place rack at x 200, y -200 mm.

Goal: the strip rotor snapped on, the two strips in opposite slots, the tube rotor left on the table and the lid
latched shut. The lid opens only by pressing its tab; the strip rotor drops over the shaft only turned to its key
and clicks on only when pushed down; the strips go in only after it is on.

The centrifuge is one loose mechkit articulation (mechkit.py, next to this file): the body with the shaft, its cross
pin and collar; the lid on its hinge and the tab on a slider on the lid; two sprung plungers in the shaft (the snap).
The rotors, strips and rack are loose compounds. The key and the snap are contact only. One declared rule
(``_lid_rule``) is the lid's catch: layout.Latch drives the lid hinge's spring target (shut and held while latched,
the pop-up spring once the tab has freed it). The arms are read through rigstate.py (the hands-off check).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import RACK, STRIP_ROTOR, STRIPS, TUBE_ROTOR, UNIT  # noqa: E402
from yam_arena.demo import ArmPlan, Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of hands off

INSTRUCTION = ("Get the mini centrifuge ready to spin the two PCR strips: tube rotor off onto the bench, strip rotor "
               "clicked on, the strips in opposite slots and the lid latched.")


# -- the parts --------------------------------------------------------------------------------------------------------
LID = mk.hinge("lid", "Y", (L.HINGE_X, 0.0, L.HINGE_Z), P.lid_shapes(), upper=L.LID_UPPER, mass=L.LID_MASS,
               spring=L.LID_SPRING, damping=L.LID_DAMP, color=L.LID_TINT)
TAB = mk.slider("tab", "-Z", P.tab_shapes(), upper=L.TAB_TRAVEL, mass=L.TAB_MASS, spring=L.TAB_SPRING,
                damping=1.0, parent="lid", color=L.LID_TINT)
PLUNGER_P = mk.slider("plunger_p", "Y", P.nose_shapes(math.pi / 2), upper=0.0, lower=-L.PLUNGER_TRAVEL,
                      mass=L.PLUNGER_MASS, spring=L.PLUNGER_SPRING, damping=L.PLUNGER_DAMP, color=L.STEEL)
PLUNGER_N = mk.slider("plunger_n", "-Y", P.nose_shapes(-math.pi / 2), upper=0.0, lower=-L.PLUNGER_TRAVEL,
                      mass=L.PLUNGER_MASS, spring=L.PLUNGER_SPRING, damping=L.PLUNGER_DAMP, color=L.STEEL)
UNIT_PART = mk.assembly(UNIT, origin=(L.UX, L.UY, L.TOP), mass=L.BODY_MASS, shapes=P.body_shapes(),
                        joints=[LID, TAB, PLUNGER_P, PLUNGER_N], color=L.WHITE, friction=L.UNIT_FRICTION,
                        tight_fit=True)
TUBE_ROTOR_PART = mk.assembly(TUBE_ROTOR, origin=(L.UX, L.UY, L.TOP + L.SEAT_Z), mass=L.TUBE_ROTOR_MASS,
                              shapes=P.tube_rotor_shapes(), color=L.TUBE_ROTOR_GREY, friction=L.ROTOR_FRICTION,
                              tight_fit=True)
STRIP_ROTOR_PART = mk.assembly(STRIP_ROTOR, origin=(L.STRIP_ROTOR_START[0], L.STRIP_ROTOR_START[1],
                                                    L.TOP + L.STRIP_ROTOR_START[2]),
                               yaw=L.STRIP_ROTOR_START_YAW, mass=L.STRIP_ROTOR_MASS, shapes=P.strip_rotor_shapes(),
                               color=L.STRIP_ROTOR_BLUE, friction=L.ROTOR_FRICTION, tight_fit=True)
RACK_PART = mk.assembly(RACK, origin=(L.RACK_X, L.RACK_Y, L.TOP), mass=L.RACK_MASS, shapes=P.rack_shapes(),
                        color=L.RACK_WHITE, friction=(0.8, 0.7), tight_fit=True)
STRIP_PARTS = tuple(mk.assembly(name, origin=(x, y, L.TOP + z), mass=L.STRIP_MASS, shapes=P.strip_shapes(),
                                color=L.STRIP_CLEAR, friction=L.STRIP_FRICTION, tight_fit=True)
                    for name, (x, y, z) in zip(STRIPS, L.STRIP_STARTS))
OBJECTS = [UNIT_PART, TUBE_ROTOR_PART, STRIP_ROTOR_PART, RACK_PART, *STRIP_PARTS]
SPECS = {o.name: o for o in OBJECTS}


# -- the one rule: the lid's catch -------------------------------------------------------------------------------------
_CATCH = L.Latch()


@mk.on_step
def _lid_rule(states) -> None:
    """Every control step: feed the catch the lid angle and the tab's travel; latched, the lid's spring target is
    -2 rad (0.30 N m holding it shut on its stop); freed, +2 rad (the pop-up spring, 0.30 N m shut, 0.10 N m at the
    75-degree back stop)."""
    try:
        u = states[UNIT]
        _CATCH.step(float(u.joints["lid"]), float(u.joints["tab"]))
    except (KeyError, AttributeError, TypeError, ValueError):
        pass  # an unreadable step keeps the last target
    mk.set_targets(UNIT_PART, "lid", _CATCH.target())


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows no lid, key, snap or slot. Offline only (never in the simulator), this model gives the
# scorer the parts' states from the arms' finger pads (two 8 x 27 x 38 mm boxes each side of the gripper's axis,
# closing along the wrist's closing axis):
# (1) the lid: freed, it opens at 2 rad/s to its stop unless a pad is on it (inside its outline and less than 12 mm
#     under its outer face), when it goes down to where no pad is in it; the tab pressed to the depth of the lowest
#     pad over it, unless that hand also rests on the shut plate; the catch is layout.Latch, as in the simulator; a
#     hand that could only be where it is by passing through the shut lid is ignored (the lid would have stopped it);
# (2) a rotor: held while a gripper is closed round its hub (20 to 23 mm between the pads, its axis within 4 mm),
#     turning with the wrist, and off the shaft only with the lid open 60 degrees or more; let go over the shaft
#     (within 2 mm, the shaft free, the lid open) it rests on the pin (8 mm proud) or, within 10 degrees of the key,
#     on the noses (4.8 mm proud); a pad pressing its hub's top pushes it down, and under 1.5 mm proud it clicks in;
#     elsewhere it falls onto the lid, the other rotor, the chamber floor or the table;
# (3) a strip: held while a gripper is closed on it end to end (66 to 72.5 mm between the pads, along it) or across
#     its cap row (4 to 7.5 mm between the pads, the closing axis across it, the TCP over its length); let go, it
#     falls into a slot of the strip rotor or the rack if it is over one, else onto whatever is under it.
# The checker's arm snapshot has no wrist turn, so the model reads each arm's closing axis from the demo's own plan at
# the step's time (``_closing_axes``); tests give it directly.
DT = 1.0 / L.CONTROL_HZ
HUB_W = 2 * L.HUB_HALF
OPEN_RATE = 2.0  # rad/s
Q_GRID = np.radians(np.arange(0.0, L.LID_UPPER_DEG + 1e-9, 0.25))
PRESS_BAND = 0.012  # a pad counts as on the lid up to 12 mm under its outer face (the 4 mm plate and 8 mm of give)
PAD_T, PAD_HW, PAD_LEN = 0.008, rigstate.FINGER_HALF_WIDTH, 0.038


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_quat(yaw: float) -> tuple:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def pad_points(tips, opening: float, axis) -> np.ndarray:
    """Points on both finger pads (a 3 x 3 x 3 grid in each), world."""
    c = np.array([float(axis[0]), float(axis[1]), 0.0])
    c /= np.linalg.norm(c) or 1.0
    w = np.array([-c[1], c[0], 0.0])
    gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
    tips = np.asarray(tips, dtype=float)
    out = []
    for side in (-1.0, 1.0):
        centre = tips + side * (gap / 2 + PAD_T / 2) * c + np.array([0.0, 0.0, PAD_LEN / 2])
        for a in (-1.0, 0.0, 1.0):
            for b in (-1.0, 0.0, 1.0):
                for h in (-1.0, 0.0, 1.0):
                    out.append(centre + a * PAD_T / 2 * c + b * PAD_HW * w + h * PAD_LEN / 2 * np.array([0, 0, 1.0]))
    return np.array(out)


class _Body:
    """A loose rotor or strip in the offline model."""

    def __init__(self, name: str, pos, yaw: float, depth: float, mode: str, rise: float = 0.0) -> None:
        self.name, self.pos, self.yaw, self.depth, self.mode, self.rise = name, np.array(pos, float), yaw, depth, \
            mode, rise
        self.holder = None
        self.off = np.zeros(3)
        self.ang0 = 0.0
        self.yaw0 = yaw
        self.vz = 0.0
        self.local = None  # a strip in a slot: (x, y, z, yaw) in the strip rotor's frame
        self.span = 0.0  # a held strip: the most the pads may open before it drops (end to end or across)


class OfflineModel:
    def __init__(self, axes_of=None) -> None:
        self.axes_of = axes_of  # step -> {"left": closing axis, "right": ...} (world vectors) or None
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.lid = 0.0
        self.tab = 0.0
        self.catch = L.Latch()
        self.plungers = 0.0
        tube = (L.UX, L.UY, L.TOP + L.SEAT_Z)
        sx, sy, sz = L.STRIP_ROTOR_START
        self.rotors = {TUBE_ROTOR: _Body(TUBE_ROTOR, tube, 0.0, L.TUBE_ROTOR_T, "shaft", 0.0),
                       STRIP_ROTOR: _Body(STRIP_ROTOR, (sx, sy, L.TOP + sz), L.STRIP_ROTOR_START_YAW,
                                          L.STRIP_ROTOR_T, "rest")}
        self.strips = {n: _Body(n, (x, y, L.TOP + z), 0.0, 0.0, "rest") for n, (x, y, z) in zip(STRIPS,
                                                                                               L.STRIP_STARTS)}

    # -- the lid --------------------------------------------------------------------------------------------------
    @staticmethod
    def _lid_frame(pts: np.ndarray, q) -> tuple[np.ndarray, np.ndarray]:
        """(u, w) of world points in the lid's frame (hinge origin, closed axes) at angle(s) q."""
        px = pts[:, 0] - (L.UX + L.HINGE_X)
        pz = pts[:, 2] - (L.TOP + L.HINGE_Z)
        q = np.atleast_1d(q)[:, None]
        return px[None, :] * np.cos(q) - pz[None, :] * np.sin(q), px[None, :] * np.sin(q) + pz[None, :] * np.cos(q)

    def _pressing(self, pts: np.ndarray, q) -> np.ndarray:
        """For each angle in q: a pad point on the lid (or its tab) at that angle."""
        if len(pts) == 0:
            return np.zeros(len(np.atleast_1d(q)), dtype=bool)
        u, w = self._lid_frame(pts, q)
        y = np.abs(pts[:, 1] - L.UY)[None, :]
        top = L.LID_Z1 - L.HINGE_Z
        plate = (u >= L.LID_X0 - L.HINGE_X) & (u <= L.LID_X1 - L.HINGE_X) & (y <= L.HY)
        tab = (u >= L.TAB_U0 - L.HINGE_X) & (u < L.LID_X0 - L.HINGE_X) & (y <= L.TAB_HY)
        depth = (w < top) & (w > top - PRESS_BAND)
        return np.any((plate | tab) & depth, axis=1)

    def _through(self, pts: np.ndarray) -> bool:
        """A hand inside the shut (under 30 degrees) lid's outline, deeper than the press band and above the deck:
        it could only be there by passing through the lid, so the lid would have stopped it. Such a hand is ignored
        for the lid, the tab and pressing a hub (the checker's arms cannot be stopped)."""
        if len(pts) == 0 or self.lid >= math.radians(30.0):
            return False
        u, w = self._lid_frame(pts, self.lid)
        u, w = u[0], w[0]
        y = np.abs(pts[:, 1] - L.UY)
        outline = (u >= L.TAB_U0 - L.HINGE_X) & (u <= L.LID_X1 - L.HINGE_X) & (y <= L.HY)
        top = L.LID_Z1 - L.HINGE_Z
        return bool(np.any(outline & (w > L.DECK - L.HINGE_Z) & (w <= top - PRESS_BAND)))

    def _step_lid(self, pts: np.ndarray, hands=None) -> None:
        """``pts``: every pad point that can act on the lid; ``hands``: the same points per hand (default one)."""
        hands = [pts] if hands is None else hands
        tab = 0.0
        top = L.LID_Z1 - L.HINGE_Z
        if self.lid < math.radians(1.0):
            for hp in hands:
                if len(hp) == 0:
                    continue
                u, w = self._lid_frame(hp, 0.0)
                u, w = u[0], w[0]
                y = np.abs(hp[:, 1] - L.UY)
                band = (w < top) & (w > top - PRESS_BAND)
                on_plate = (u >= L.LID_X0 - L.HINGE_X) & (u <= L.LID_X1 - L.HINGE_X) & (y <= L.HY) & band
                if np.any(on_plate):
                    continue  # this hand rests on the shut plate, which holds it up: it cannot push the tab down
                over = (u >= L.TAB_U0 - L.HINGE_X) & (u <= L.TAB_U1 - L.HINGE_X) & (y <= L.TAB_HY) & band
                if np.any(over):
                    tab = max(tab, float(np.clip(top - w[over].min(), 0.0, L.TAB_TRAVEL)))
        self.tab = tab
        self.catch.step(self.lid, self.tab)
        if self.catch.latched:
            self.lid = 0.0
            return
        pressing = self._pressing(pts, Q_GRID)
        i = int(np.searchsorted(Q_GRID, self.lid - 1e-9))
        i = min(i, len(Q_GRID) - 1)
        if pressing[i]:
            while i > 0 and pressing[i]:
                i -= 1
        else:
            top = int(np.searchsorted(Q_GRID, min(self.lid + OPEN_RATE * DT, L.LID_UPPER) + 1e-9)) - 1
            while i < top and not pressing[i + 1]:
                i += 1
        self.lid = float(Q_GRID[i])

    # -- rotors and strips ------------------------------------------------------------------------------------------
    def _support(self, b: _Body) -> tuple[str, float]:
        """Where a let-go rotor comes to rest: ("shaft", rise), ("lid", origin z) or ("rest", origin z)."""
        d = math.hypot(b.pos[0] - L.UX, b.pos[1] - L.UY)
        over_unit = abs(b.pos[0] - L.UX) <= L.HX and abs(b.pos[1] - L.UY) <= L.HY
        if over_unit and self.lid < math.radians(60.0):
            return "lid", L.TOP + L.LID_Z1 + b.depth  # on the lid (not open far enough to get in)
        other = [o for o in self.rotors.values() if o is not b and o.mode == "shaft"]
        if other and d <= L.TUBE_ROTOR_R:
            return "rest", other[0].pos[2] + L.HUB_H + b.depth  # on the rotor still on the shaft
        if d <= 0.002:
            keyed = min(abs(wrap(b.yaw)), abs(wrap(b.yaw - math.pi))) <= math.radians(L.KEY_TOL_DEG)
            return "shaft", (L.NOSE_REST if keyed else L.PIN_REST)
        if over_unit:
            return "rest", L.TOP + L.BOWL_FLOOR + b.depth
        return "rest", L.TOP + b.depth

    def _strip_support(self, s: _Body) -> tuple[str, float, tuple | None]:
        if abs(s.pos[0] - L.UX) <= L.HX and abs(s.pos[1] - L.UY) <= L.HY and self.lid < math.radians(60.0):
            return "rest", L.TOP + L.LID_Z1, None  # on the lid
        rot = self.rotors[STRIP_ROTOR]
        c, sn = math.cos(rot.yaw), math.sin(rot.yaw)
        d = s.pos - rot.pos
        lx, ly = c * d[0] + sn * d[1], -sn * d[0] + c * d[1]
        turn = abs(wrap(2 * (s.yaw - rot.yaw))) / 2
        if rot.mode in ("shaft", "rest") and turn <= 0.15:
            for y in L.SLOT_Y:
                if abs(ly - y) <= L.SLOT_W / 2 - L.TUBE_R and abs(lx) <= (L.SLOT_LX - L.STRIP_L) / 2 + 0.0015:
                    if s.pos[2] >= rot.pos[2] + L.SEATED_Z - 1e-6:
                        return "slot", rot.pos[2] + L.SEATED_Z, (lx, ly, L.SEATED_Z, s.yaw - rot.yaw)
        if rot.mode in ("shaft", "rest") and abs(lx) <= L.STRIP_ROTOR_X / 2 and abs(ly) <= L.STRIP_ROTOR_Y / 2:
            return "rest", rot.pos[2], None
        for col in L.RACK_COLS:
            if abs(s.pos[1] - (L.RACK_Y + L.rack_col_y(col))) <= L.RACK_HOLE_W / 2 - L.TUBE_R and \
                    abs(s.pos[0] - L.RACK_X) <= 0.003 and abs(wrap(2 * s.yaw)) / 2 <= 0.15:
                return "rest", L.TOP + L.RACK_FLOOR, None
        if abs(s.pos[0] - L.RACK_X) <= L.RACK_LX / 2 and abs(s.pos[1] - L.RACK_Y) <= L.RACK_LY / 2:
            return "rest", L.TOP + L.RACK_H, None
        tube = self.rotors[TUBE_ROTOR]
        if math.hypot(*(s.pos[:2] - tube.pos[:2])) <= L.TUBE_ROTOR_R:
            return "rest", tube.pos[2], None
        if abs(s.pos[0] - L.UX) <= L.HX and abs(s.pos[1] - L.UY) <= L.HY:
            return "rest", L.TOP + L.BOWL_FLOOR, None
        return "rest", L.TOP, None

    @staticmethod
    def _fall(b: _Body, rest_z: float) -> bool:
        """One step of a free fall toward rest_z; True once there."""
        if b.pos[2] <= rest_z + 1e-9:
            b.pos[2] = max(b.pos[2], rest_z) if b.pos[2] > rest_z - 0.002 else b.pos[2]
            b.vz = 0.0
            return True
        b.vz += L.G * DT
        b.pos[2] = max(rest_z, b.pos[2] - b.vz * DT)
        if b.pos[2] <= rest_z + 1e-9:
            b.vz = 0.0
            return True
        return False

    def _hold(self, b: _Body, tcp, ang: float) -> None:
        c, s = math.cos(ang - b.ang0), math.sin(ang - b.ang0)
        o = b.off
        b.pos = np.asarray(tcp, float) + np.array([c * o[0] - s * o[1], s * o[0] + c * o[1], o[2]])
        b.yaw = b.yaw0 + (ang - b.ang0)

    def step(self, arms) -> None:
        self.n += 1
        read = rigstate.read_arms(arms)
        axes = self.axes_of(self.n) if self.axes_of is not None else None
        hands = {}
        if read is not None:
            for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
                axis = None if axes is None else axes.get(side)
                if axis is None:
                    axis = (0.0, 1.0, 0.0)
                hands[side] = (np.asarray(tcp, float), np.asarray(tips, float), float(opening),
                               math.atan2(axis[1], axis[0]), pad_points(tips, opening, axis))
        free = [h[4] for h in hands.values() if not self._through(h[4])]
        pts = np.concatenate(free) if free else np.zeros((0, 3))
        self._step_lid(pts, free)
        lid_open = self.lid >= math.radians(60.0)
        # rotors
        for b in self.rotors.values():
            if b.holder is not None:
                h = hands.get(b.holder)
                gap = None if h is None else h[2] * rigstate.FULL_GAP
                if gap is None or gap > HUB_W + 0.0015:
                    b.holder = None
                    b.mode = "fall"
                else:
                    self._hold(b, h[0], h[3])
                    continue
            for side, (tcp, tips, opening, ang, _) in hands.items():
                gap = opening * rigstate.FULL_GAP
                axis_ok = abs(wrap(4 * (ang - b.yaw))) / 4 <= math.radians(12.0)
                near = math.hypot(*(tcp[:2] - b.pos[:2])) <= 0.004
                height = b.pos[2] + 0.0005 <= tips[2] <= b.pos[2] + L.HUB_H - 0.004
                reachable = b.mode != "shaft" or lid_open
                if HUB_W - 0.006 <= gap <= HUB_W + 0.001 and axis_ok and near and height and reachable:
                    b.holder, b.mode = side, "held"
                    off = b.pos - tcp
                    c, s = math.cos(-ang), math.sin(-ang)
                    b.off = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1], off[2]])
                    b.ang0, b.yaw0 = 0.0, b.yaw - ang
                    self._hold(b, tcp, ang)
                    break
            if b.holder is not None:
                continue
            if b.mode == "fall":
                where, value = self._support(b)
                if where == "shaft":
                    rest = L.TOP + L.SEAT_Z + value
                    if self._fall(b, rest):
                        b.mode, b.rise = "shaft", value
                        b.pos[:2] = (L.UX, L.UY)
                else:
                    if where == "lid":
                        b.pos[2] = max(b.pos[2], value)  # a hand that went through the shut lid: back on top
                    if self._fall(b, value):
                        b.mode = "rest"
            elif b.mode == "shaft":
                keyed = min(abs(wrap(b.yaw)), abs(wrap(b.yaw - math.pi))) <= math.radians(L.KEY_TOL_DEG)
                floor = 0.0 if keyed else L.PIN_REST
                top = L.TOP + L.SEAT_Z + b.rise + L.HUB_H
                pressed = False
                if len(pts):
                    c, s = math.cos(b.yaw), math.sin(b.yaw)
                    d = pts - b.pos
                    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
                    on = (np.abs(lx) <= L.HUB_HALF + 0.001) & (np.abs(ly) <= L.HUB_HALF + 0.001) & \
                         (pts[:, 2] < top) & (pts[:, 2] > top - 0.02)
                    if np.any(on):
                        pressed = True
                        b.rise = max(floor, min(b.rise, float(pts[on, 2].min()) - L.HUB_H - L.TOP - L.SEAT_Z))
                if keyed and b.rise <= 0.0015:
                    b.rise = 0.0
                elif keyed and not pressed and 0.0035 < b.rise < L.NOSE_REST:
                    b.rise = L.NOSE_REST
                b.pos = np.array([L.UX, L.UY, L.TOP + L.SEAT_Z + b.rise])
        on_shaft = [b for b in self.rotors.values() if b.mode == "shaft"]
        self.plungers = -0.00075 if any(0.0002 < b.rise < L.NOSE_REST - 1e-4 for b in on_shaft) else 0.0
        # strips
        rot = self.rotors[STRIP_ROTOR]
        for s in self.strips.values():
            if s.holder is not None:
                h = hands.get(s.holder)
                gap = None if h is None else h[2] * rigstate.FULL_GAP
                if gap is None or gap > s.span:
                    s.holder, s.mode = None, "fall"
                else:
                    self._hold(s, h[0], h[3])
                    continue
            if s.mode == "slot":
                lx, ly, lz, lyaw = s.local
                c, sn = math.cos(rot.yaw), math.sin(rot.yaw)
                s.pos = rot.pos + np.array([c * lx - sn * ly, sn * lx + c * ly, lz])
                s.yaw = rot.yaw + lyaw
            for side, (tcp, tips, opening, ang, _) in hands.items():
                gap = opening * rigstate.FULL_GAP
                along = np.array([math.cos(s.yaw), math.sin(s.yaw)])
                d = tcp[:2] - s.pos[:2]
                du, dw = abs(float(d @ along)), abs(float(d @ np.array([-along[1], along[0]])))
                height = s.pos[2] + 0.008 <= tips[2] <= s.pos[2] + L.STRIP_H - 0.001
                ends = (L.STRIP_L - 0.006 <= gap <= L.STRIP_L + 0.0005 and abs(wrap(2 * (ang - s.yaw))) / 2 <= 0.2
                        and du <= 0.004 and dw <= 0.003)
                across = (L.STRIP_W - 0.003 <= gap <= L.STRIP_W + 0.0005 and
                          abs(wrap(2 * (ang - s.yaw - math.pi / 2))) / 2 <= 0.2 and du <= L.STRIP_L / 2 - 0.004 and
                          dw <= 0.002)
                if (ends or across) and height:
                    s.holder, s.mode, s.local = side, "held", None
                    s.span = L.STRIP_L + 0.001 if ends else L.STRIP_W + 0.0015
                    off = s.pos - tcp
                    c, sn = math.cos(-ang), math.sin(-ang)
                    s.off = np.array([c * off[0] - sn * off[1], sn * off[0] + c * off[1], off[2]])
                    s.ang0, s.yaw0 = 0.0, s.yaw - ang
                    self._hold(s, tcp, ang)
                    break
            if s.holder is None and s.mode == "fall":
                where, z, loc = self._strip_support(s)
                if self._fall(s, z):
                    s.mode = where
                    s.local = loc

    def states(self) -> dict:
        joints = {"lid": self.lid, "tab": self.tab, "plunger_p": self.plungers, "plunger_n": self.plungers}
        out = {UNIT: mk.state_from_poses(UNIT_PART, (L.UX, L.UY, L.TOP, 0.0, 0.0, 0.0, 1.0), joints=joints,
                                         source="offline"),
               RACK: mk.state_from_poses(RACK_PART, (L.RACK_X, L.RACK_Y, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                         source="offline")}
        for b in self.rotors.values():
            out[b.name] = mk.state_from_poses(SPECS[b.name], (*b.pos, *yaw_quat(b.yaw)), source="offline")
        for s in self.strips.values():
            out[s.name] = mk.state_from_poses(SPECS[s.name], (*s.pos, *yaw_quat(s.yaw)), source="offline")
        return out


def _closing_axes(step: int) -> dict:
    """Offline only: each arm's closing axis (world) at control step ``step`` (time step / 20 s), from the demo's
    plan as ``demo()`` would give it now (an idle arm: y)."""
    plan = demo()
    t = step * DT
    out = {}
    for side, base in (("left", (0.0, 0.305, 0.75)), ("right", (0.0, -0.305, 0.75))):
        if side not in plan:
            out[side] = (0.0, 1.0, 0.0)
            continue
        key = (side, tuple((w.duration, w.pos, w.finger, w.yaw) for w in plan[side]))
        ap = _ARM_PLANS.get(key)
        if ap is None:
            ap = _ARM_PLANS.setdefault(key, ArmPlan(plan[side], base))
        rot = np.asarray(ap.target(t)[1], dtype=float)
        out[side] = tuple(rot[:, 0])
    return out


_ARM_PLANS: dict = {}
_OFFLINE = OfflineModel(_closing_axes)


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(UNIT), "source", "sim") != "sim"


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
    _CATCH.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The strip rotor clicked on, both strips in opposite slots, the lid latched, the tube rotor on the table,
    hands off for 1 s and no zero event; held 1 s. This runs first every control step, so it also feeds the tracker
    (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    if no_simulator(parts):
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, "rule_catch": _CATCH.snapshot(), **result.details}
    for n in S.LOOSE:
        try:
            row[f"{n}_origin"] = parts[n].origin
        except (AttributeError, KeyError, TypeError):
            row[f"{n}_origin"] = None
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(UNIT), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a rotor or strip fell off the table, the unit tilted over 10 degrees (a slide alone does not count), the
    latched lid was forced, the lid was freed again after it had latched shut, hands were not off for the last 1 s, or the lid
    is latched over an unseated rotor or an unbalanced load; else 1.0 rotor clicked on, strips in opposite slots, lid
    latched, tube rotor on the table; 0.75 the same with the lid open; 0.5 rotor clicked on and tube rotor stored, the
    strips not balanced; 0.25 tube rotor stored and the strip rotor on the shaft, not clicked. The last step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm under the TCP. yaw 0 closes the fingers along y; +pi/2 (right arm) and -pi/2 (left arm) along x.
#
# Clearance (2026-11-19, second fix round): every gripper housing keeps 10 mm or more from the open lid (at its stop,
# while it springs open and while it is swung shut), from the unit's body and from the other arm's housing at every
# step, by the YAM collision capsules (tools/offline/yamkin.py: the gripper link and each finger's carriage capsule;
# the pad capsules are the parts meant to touch). That is the test test_housings_keep_10_mm_from_the_lid_unit_and_
# each_other. The rules it forced: (1) over the open chamber nothing closes along x: turned to yaw +-90 the housing
# reaches 53 to 67 mm along the closing axis (102 mm in all, research/sim_capabilities.md) and the far finger's
# carriage capsule went 23 mm into the open lid while a strip was held end to end over the shaft; so both strips are
# pinched across their cap row near their front end (yaw 0) and both rotors are handled at yaw 0; (2) over the
# chamber the fingers open only 36 mm (7 mm clear each side of a hub), not 94: open, the carriages reach the lid's
# side skirts; (3) the freed lid is not let fly up through the housing: the block backs off forward and up in 0.6 s,
# so the housing is ahead of and above the tab's path as the lid springs up (31 mm at the stated 0.7 s opening, 18 mm
# if it opened 1.5 times as fast); (4) the lid is swung shut by the block's edge 90 mm in front of the hinge (60 mm
# put the housing into the lid's upper part); (5) in the rack a strip is pinched behind its middle and turned end for
# end on the way (pinched in front of it, the right arm's link4 went into the open lid's corner).
PAD_TIP = 0.013
OPEN = -0.04695  # 94 mm
CLOSED = 0.0
HUB_GRIP = -(HUB_W - 0.002) / 2  # 20 mm commanded on the 22 mm hub: 5 N a finger (the 5000 N/m finger drive)
HUB_OPEN = -0.036 / 2  # 36 mm: 7 mm clear each side of a 22 mm hub
STRIP_PINCH = -(L.STRIP_W - 0.002) / 2  # 5 mm on the 7 mm cap row: 5 N a finger
STRIP_OPEN = -0.022 / 2  # 22 mm: 7.5 mm clear each side of the cap row
STRIP_OFF = 0.030  # a strip is pinched 30 mm from its middle: at its back end in the rack, at its front over the rotor
YAW_FLIP = math.pi  # the right wrist turned end for end (joint 6 = joint 1 - yaw: 0.12 rad or more from its limit)
BLOCK_HALF = PAD_T  # closed, the two pads are a block 16 mm across the closing axis and 27 mm along the other
YAW_X_R, YAW_X_L = math.pi / 2, -math.pi / 2


def _w(d, x, y, h, finger, label, yaw):
    """A waypoint with the TCP ``h`` over the table top."""
    return Waypoint(d, (float(x), float(y), L.TOP + float(h)), finger, label, yaw)


def _tips(d, x, y, tip_h, finger, label, yaw):
    return _w(d, x, y, tip_h + PAD_TIP, finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def _wait_until(plan, t_abs, label):
    """Hold the last waypoint until ``t_abs`` (s from the start)."""
    last = plan[-1]
    d = t_abs - _duration(plan)
    if d > 0:
        plan.append(Waypoint(d, last.pos, last.finger, label, last.yaw))


# The tab: a 14 x 20 mm lip in front of the lid's front edge. The right hand's closed block (16 mm along x) is centred
# 11 mm in front of that edge, so it covers 11 mm of the tab and stays 3 mm clear of the lid itself.
TAB_X = L.UX + L.LID_X0 - 0.011
PRESS_OVER = 0.010  # the block commanded 10 mm under the tab's top (3 mm of travel, then the arm's give)
# The tube rotor's hub is pinched 3 mm over its underside (the pads 3 mm clear of its disc), 20 mm commanded.
HUB_TIP = 0.003
LIFT_H = 0.165  # carrying the tube rotor: the TCP 165 mm up (its disc 135 mm up, clear of the front panel and deck)
CARRY_H = 0.150  # carrying the strip rotor in: its plate's underside 114 mm up, 22 mm over the deck, and its far
# edge (40 mm behind the shaft) 15 mm in front of the open lid's skirts
# Keyed and let go 0.7 mm over the noses; then pressed with the closed block 10 mm under the seated hub's top.
DROP_RISE = L.NOSE_REST + 0.0007
SEAT_OVER = 0.010
# A strip is pinched across its cap row with the pad tips 1.5 mm over the rack's top (14.5 mm over the strip's
# underside) and let go with its underside 5 mm over a slot floor (the pad tips then 2.5 mm over the rotor's top).
STRIP_TIP = L.RACK_H + 0.0015 - L.RACK_FLOOR
STRIP_DROP = 0.005
# In the rack a strip is pinched 30 mm behind its middle (yaw 0): pinched in front of its middle, the TCP at x 170 mm,
# the right arm's link4 capsule went 11 mm into the open lid's corner over the column at y -178. On the way to the
# chamber the wrist turns it end for end (two quarter turns, so the turn always goes the way joint 6 can follow), and
# over the rotor the TCP is 30 mm in front of the strip's middle.
# Into the chamber: the strip crosses the unit's side with its middle 40 mm in front of its slot and its underside
# 15 mm over the front panel (STRIP_HIGH_TIP), comes across in line with its slot and in to 10 mm short of the slot's
# middle at that height (its back end then 8 mm past the front panel), drops to 4 mm over the rotor's top, slides the
# last 10 mm with its underside 1 mm over the rotor's top and only then goes down into the slot. Pinched 30 mm in
# front of its middle, the pads (27 mm along x) stay 5 mm or more in front of the hub's outline; the housing (65 mm
# across x) stays over the front half of the chamber, 40 mm or more from the open lid.
STRIP_HIGH_TIP = L.COWL_TOP + 0.015 + STRIP_TIP
STRIP_FRONT = 0.040
STRIP_IN = 0.010
STRIP_OVER = 0.004
STRIP_SLIDE = 0.001
SLOTS_USED = (1, 2)  # an opposite pair, 19 mm each side of the hub
# Freeing the lid: the closed block (yaw +90, 16 mm along x) presses the tab 10 mm past its top, holds, then backs
# off forward and up in one quick move (RETREAT_X forward, RETREAT_UP over the lid's top, RETREAT_S): the lid is
# held down until the block's +x face passes the tab's front edge and then springs up behind the hand, whose housing
# is by then ahead of and above the edge's path. The hand cannot simply rise off the tab: the lid's edge sweeps a
# 140 mm arc round the hinge up to 239 mm over the table, and this arm reaches only 200 mm (TCP) over the midline.
RETREAT_X = 0.060
RETREAT_UP = 0.060
RETREAT_S = 0.6
# Shutting the lid: the closed block at yaw 0 (27 mm along x) rides the lid's top with its lower -x edge 90 mm in
# front of the hinge (34 mm behind the plate's front edge) and 75 mm to the right of the middle (its outer pad 4 mm
# inside the plate's edge), commanded 6 mm into it, from the 75-degree stop down to shut, then 10 mm deeper (the
# catch). It comes in level from 15 mm behind the lid. Nearer the hinge the housing (40 mm each side of the TCP along
# x, from 30 mm over the TCP up) runs into the lid's upper part while it is steep (60 mm, the first build: 14 mm into
# it); further out the start is beyond this arm's reach (TCP 205 mm over the table there, joint 4 within 3.5 degrees
# of its limit at 90 mm).
CLOSE_RHO = 0.090
CLOSE_HALF = PAD_HW
CLOSE_INTO = 0.006
CLOSE_STANDOFF = 0.015
CLOSE_Y = -0.075
LATCH_OVER = 0.010
PARK_L = (0.140, 0.300, 0.160)
PARK_R = (0.150, -0.280, 0.160)


def _edge_tcp(q_deg: float, w_off: float) -> tuple[float, float]:
    """The right TCP (world x, height) that puts the closed block's lower -x edge (yaw 0) on the lid ``w_off``
    outside its top face (negative: into it), CLOSE_RHO in front of the hinge, the lid open ``q_deg``."""
    x, z = L.lid_point(CLOSE_RHO, math.radians(q_deg), L.LID_Z1 - L.HINGE_Z + w_off)
    return L.UX + x + CLOSE_HALF, z + PAD_TIP


def right_moves(t_side: float, strips=STRIPS, slots=SLOTS_USED, lid_from=L.LID_UPPER_DEG, skip_close=False):
    """The right arm: press the lid's tab with a closed block and back off forward and up (the lid springs open
    behind it); pinch each strip across its cap row in the rack and, from beside the chamber at
    ``t_side`` (once the strip rotor is on and the left hand gone), drop it in its slot; swing the lid shut from
    behind and press it till it latches; aside."""
    rack_top = L.RACK_H
    plan = [
        _tips(2.0, TAB_X, L.UY, L.LID_Z1 + 0.030, CLOSED, "over the lid's tab, fingers closed into a block", YAW_X_R),
        _tips(1.5, TAB_X, L.UY, L.LID_Z1 + 0.003, CLOSED, "down to just over the tab", YAW_X_R),
        _tips(1.5, TAB_X, L.UY, L.LID_Z1 - PRESS_OVER, CLOSED, "press the tab: the catch lets go", YAW_X_R),
        _tips(0.5, TAB_X, L.UY, L.LID_Z1 - PRESS_OVER, CLOSED, "hold", YAW_X_R),
        _tips(RETREAT_S, TAB_X - RETREAT_X, L.UY, L.LID_Z1 + RETREAT_UP, CLOSED, "back off forward and up: the lid "
              "springs open behind the block", YAW_X_R),
        _tips(1.5, 0.110, L.UY - 0.220, 0.120, STRIP_OPEN, "aside, clear of the left hand", 0.0),
    ]
    first = True
    flip, half = YAW_FLIP, YAW_FLIP / 2
    for name, slot in zip(strips, slots):
        sx, sy, sz = L.STRIP_STARTS[STRIPS.index(name)]
        tip = sz + STRIP_TIP
        px = sx + STRIP_OFF
        pick = [
            _tips(2.0, px, sy, rack_top + 0.050, STRIP_OPEN, f"over the back end of {name} in the rack", 0.0),
            _tips(1.5, px, sy, tip, STRIP_OPEN, "down either side of its cap row", 0.0),
            _tips(1.5, px, sy, tip, STRIP_PINCH, "pinch it across its cap row", 0.0),
            _tips(1.5, px, sy, rack_top + 0.050, STRIP_PINCH, "lift it out", 0.0),
        ]
        if first:
            _wait_until(plan, t_side - _duration(pick) - 0.5, "wait for the strip rotor")
        plan += pick
        if first:
            _wait_until(plan, t_side, "wait: the left hand presses the rotor on and leaves")
        first = False
        ty = L.UY + L.SLOT_Y[slot]
        drop_tip = L.SEAT_Z + L.SEATED_Z + STRIP_DROP + STRIP_TIP
        over_tip = L.SEAT_Z + STRIP_OVER + STRIP_TIP
        slide_tip = L.SEAT_Z + STRIP_SLIDE + STRIP_TIP
        clear_tip = L.SEAT_Z + L.SEATED_Z + L.STRIP_H + 0.003
        front, inner, home = (L.UX - d - STRIP_OFF for d in (STRIP_FRONT, STRIP_IN, 0.0))
        side_y = L.UY - 0.120
        plan += [
            _tips(1.0, (px + front) / 2, (sy + side_y) / 2, STRIP_HIGH_TIP, STRIP_PINCH,
                  "up toward the unit's front, turning it a quarter turn", half),
            _tips(1.0, front, side_y, STRIP_HIGH_TIP, STRIP_PINCH, "to the chamber's side, turned end for end", flip),
            _tips(1.5, front, ty, STRIP_HIGH_TIP, STRIP_PINCH, f"across, in line with slot {slot}", flip),
            _tips(1.0, inner, ty, STRIP_HIGH_TIP, STRIP_PINCH, "in, 10 mm short of the slot's middle", flip),
            _tips(1.0, inner, ty, over_tip, STRIP_PINCH, "down to 4 mm over the rotor's top", flip),
            _tips(1.0, home, ty, slide_tip, STRIP_PINCH, "along over the slot's middle", flip),
            _tips(1.0, home, ty, drop_tip, STRIP_PINCH, "lower it into the slot", flip),
            _tips(0.3, home, ty, drop_tip, STRIP_PINCH, "settle", flip),
            _tips(1.0, home, ty, drop_tip, STRIP_OPEN, "let go: it drops onto the slot floor", flip),
            _tips(0.8, home, ty, clear_tip, STRIP_OPEN, "up just clear of its caps", flip),
            _tips(0.8, home, ty, STRIP_HIGH_TIP, STRIP_OPEN, "straight up", flip),
            _tips(1.5, front, side_y, STRIP_HIGH_TIP, STRIP_OPEN, "forward and out to the side", flip),
            _tips(0.8, front, side_y, STRIP_HIGH_TIP, STRIP_OPEN, "turn the wrist back a quarter turn", half),
        ]
    if skip_close:
        plan.append(_w(2.0, *PARK_R, OPEN, "aside", 0.0))
        return plan
    x0, h0 = _edge_tcp(lid_from, -CLOSE_INTO)
    xs, _ = _edge_tcp(lid_from, 0.0)
    xs += CLOSE_STANDOFF
    plan += [
        _w(2.0, xs, CLOSE_Y - 0.090, h0, CLOSED, "beside the unit behind the open lid, fingers closed into a block",
           0.0),
        _w(1.5, xs, CLOSE_Y, h0, CLOSED, "in behind the lid", 0.0),
        _w(1.0, x0, CLOSE_Y, h0, CLOSED, "onto the lid's back, 90 mm from the hinge", 0.0),
    ]
    steps = 10
    for k in range(1, steps + 1):
        q = lid_from * (1.0 - k / steps)
        x, h = _edge_tcp(q, -CLOSE_INTO)
        plan.append(_w(0.7, x, CLOSE_Y, h, CLOSED, f"swing it shut ({q:.0f} degrees)", 0.0))
    x, h = _edge_tcp(0.0, -LATCH_OVER)
    plan += [
        _w(1.5, x, CLOSE_Y, h, CLOSED, "press it down: the catch clicks", 0.0),
        _w(0.5, x, CLOSE_Y, h, CLOSED, "hold", 0.0),
        _w(1.5, x - 0.020, CLOSE_Y, h + 0.060, CLOSED, "up off the lid", 0.0),
        _w(2.0, *PARK_R, CLOSED, "aside", 0.0),
    ]
    return plan


def left_moves(pick_yaw=YAW_X_L, enter_at=9.5, press=True):
    """The left arm: wait clear of the lid while it springs open; pinch the tube rotor's hub, lift it off past the
    snap, set it on the table; pinch the strip rotor's hub with the wrist turned ``pick_yaw``, turn it back a quarter
    turn on the way (the rotor turns with it, onto the key), lower it over the shaft and let go (it drops onto the
    noses); press its hub down with a closed block till it clicks; aside.

    Everything over the open chamber is done with the wrist at yaw 0 (fingers closing along y): turned to yaw +-90
    the gripper housing (102 mm along the closing axis, 60 mm of it on one side) reaches over the shaft into the
    open lid. That is what knocked the unit over on the first GPU run (the strip rotor brought over the shaft at
    yaw -90, 2026-11-19). Over the chamber the fingers open only 36 mm (HUB_OPEN): opened to 94 mm their carriages
    reach the lid's side skirts. ``pick_yaw`` 0 (the offkey variant) leaves the rotor a quarter turn off its key."""
    tx, ty = L.TUBE_STORE
    sx, sy, sz = L.STRIP_ROTOR_START
    hub_tip = L.SEAT_Z + HUB_TIP
    plan = [_w(2.5, 0.120, 0.220, 0.140, HUB_OPEN, "wait clear of the lid's sweep", 0.0)]
    _wait_until(plan, enter_at, "wait for the lid to open")
    plan += [
        _tips(1.0, L.UX - 0.025, L.UY + 0.040, L.SEAT_Z + L.HUB_H + 0.030, HUB_OPEN, "in past the lid's side skirt",
              0.0),
        _tips(1.0, L.UX, L.UY, L.SEAT_Z + L.HUB_H + 0.030, HUB_OPEN, "over the tube rotor's hub", 0.0),
        _tips(1.5, L.UX, L.UY, hub_tip, HUB_OPEN, "down round the hub", 0.0),
        _tips(1.5, L.UX, L.UY, hub_tip, HUB_GRIP, "pinch the hub", 0.0),
        _w(1.5, L.UX, L.UY, LIFT_H, HUB_GRIP, "lift it straight off, past the snap", 0.0),
        _w(1.2, L.UX - 0.030, L.UY + 0.020, LIFT_H, HUB_GRIP, "forward out of the chamber", 0.0),
        _w(1.8, tx, ty, LIFT_H, HUB_GRIP, "over the bench", 0.0),
        _tips(1.5, tx, ty, L.TUBE_ROTOR_T + 0.003 + HUB_TIP, HUB_GRIP, "lower it to 3 mm over the table", 0.0),
        _tips(1.5, tx, ty, L.TUBE_ROTOR_T + 0.003 + HUB_TIP, HUB_OPEN, "let go", 0.0),
        _w(1.2, tx, ty, 0.080, HUB_OPEN, "up", 0.0),
        _w(2.0, sx, sy, 0.080, HUB_OPEN, "over the strip rotor's hub, the wrist turned a quarter turn", pick_yaw),
        _tips(1.2, sx, sy, sz + HUB_TIP, HUB_OPEN, "down round its hub", pick_yaw),
        _tips(1.5, sx, sy, sz + HUB_TIP, HUB_GRIP, "pinch the hub", pick_yaw),
        _w(1.5, sx, sy, CARRY_H, HUB_GRIP, "lift it", pick_yaw),
        _w(2.0, L.UX - 0.020, sy, CARRY_H, HUB_GRIP, "in front of the open lid, turning it a quarter turn to the key",
           0.0),
        _w(1.5, L.UX - 0.020, L.UY + 0.040, CARRY_H, HUB_GRIP, "in past the lid's side skirt", 0.0),
        _w(1.0, L.UX, L.UY, CARRY_H, HUB_GRIP, "over the shaft", 0.0),
    ]
    drop_tip = L.SEAT_Z + DROP_RISE + HUB_TIP
    plan += [
        _tips(1.5, L.UX, L.UY, drop_tip, HUB_GRIP, "lower it over the shaft", 0.0),
        _tips(0.5, L.UX, L.UY, drop_tip, HUB_GRIP, "settle", 0.0),
        _tips(1.5, L.UX, L.UY, drop_tip, HUB_OPEN, "let go: it rests on the snap's noses", 0.0),
        _tips(1.0, L.UX, L.UY, L.SHAFT_TOP + 0.025, HUB_OPEN, "up", 0.0),
        _tips(1.5, L.UX, L.UY, L.SHAFT_TOP + 0.025, CLOSED, "close the fingers into a block", 0.0),
    ]
    if press:
        plan += [
            _tips(1.5, L.UX, L.UY, L.SHAFT_TOP + DROP_RISE + 0.002, CLOSED, "down onto the hub", 0.0),
            _tips(2.0, L.UX, L.UY, L.SHAFT_TOP - SEAT_OVER, CLOSED, "push it down: it clicks on", 0.0),
            _tips(0.5, L.UX, L.UY, L.SHAFT_TOP - SEAT_OVER, CLOSED, "hold", 0.0),
        ]
    plan += [
        _w(1.5, L.UX, L.UY, 0.150, CLOSED, "up", 0.0),
        _w(0.7, L.UX - 0.040, L.UY, 0.155, CLOSED, "forward, away from the open lid", 0.0),
        _w(1.0, 0.110, 0.120, 0.160, CLOSED, "back toward its own side", 0.0),
        _w(1.5, *PARK_L, CLOSED, "aside", 0.0),
    ]
    return plan


# Wrong ways (counterfactuals, run on the GPU for the README; the offline model predicts them):
#   nopress    the strip rotor let go on the noses and never pushed: not clicked, so the lid latches on a loose
#              rotor (0)
#   offkey     the strip rotor lowered a quarter turn off its key: it stays on the pin, 8 mm proud (0)
#   onestrip   only one strip loaded: the lid latches on an unbalanced load (0)
#   openlid    everything but the lid: left open (0.75)
VARIANTS = ("nopress", "offkey", "onestrip", "openlid")
SIDE_AFTER_BLOCK = 6.0  # the right hand heads for the chamber's side 6 s after the left closes its block


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    left = left_moves(pick_yaw=0.0 if variant == "offkey" else YAW_X_L, press=variant != "nopress")
    t_side = _until(left, "close the fingers into a block") + SIDE_AFTER_BLOCK
    strips = STRIPS[:1] if variant == "onestrip" else STRIPS
    right = right_moves(t_side, strips=strips, skip_close=variant == "openlid")
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo): nopress, offkey, onestrip,
    openlid."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
