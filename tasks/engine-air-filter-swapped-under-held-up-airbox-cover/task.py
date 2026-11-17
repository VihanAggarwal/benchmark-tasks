"""On the station table, real size, an engine bay's air cleaner: a lower case (300 x 240 mm, long side along x, centred
at x 360 mm, y 10 mm, rim 75 mm up) on a steel stand, with a seat pocket 30 mm deep (6 mm of play round the filter),
two slots in its rear wall and two over-centre spring clips on its front wall, all closed. Its polypropylene
cover (40 mm deep, 0.28 kg) sits on the rim, tabs in the slots, a catch lug over each clip's hook and a lift lug on its
front edge, and hangs on a flexible intake duct (a spring of 0.1 N/cm) from a fixed throttle-body stub. In the seat is
the old filter (240 x 173 x 49 mm, a 20 mm frame). To the right,
on a shop mat, lies the new one turned a quarter turn; a second mat is the drop zone for the old filter. A battery
(230 x 175 x 225 mm) and the radiator support's top edge stand where they do in an engine bay.

Goal: old filter out on its mat, new filter flat in the seat with its seal flat all round, the
cover down with its tabs in the slots, both clips closed, hands off. The cover can only be lifted with the clips
released and has to be held out of the way while the filter changes: the duct pulls it back down.

The airbox (clips are two hinged levers, a detent rule), the cover (its duct is three sprung sliders dx, dy, dz) and the
filters (compound bodies) are mechkit parts (mechkit.py, next to this file). One declared rule: the clips' over-centre
detent (``_clip_detent``, a mechkit on_step callback), notches at closed and at released. The arms are read through
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import AIRBOX, COVER, NEW, OLD  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = (
    "Change the engine air filter. The airbox's two clips are closed. Release both, hold the cover up out of the way, "
    "lift the old filter out and set it on its mat, put the new filter in its seat with its seal flat all round, lower "
    "the cover so its rear tabs go into the slots, and close both clips. Do not pull the cover off its intake duct or "
    "let a filter fall off the table."
)


# -- the parts --------------------------------------------------------------------------------------------------------
def bx(x0, x1, y0, y1, z0, z1, color=None):
    """A box from its extents (world x, y; z above the table top)."""
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def airbox_shapes():
    c, st = L.CASE_GREY, L.STAND_GREY
    x0, x1, y0, y1 = L.CASE_X0, L.CASE_X1, L.CASE_Y0, L.CASE_Y1
    px0, px1, py0, py1 = L.POCKET_X0, L.POCKET_X1, L.POCKET_Y0, L.POCKET_Y1
    slot_z0 = L.RIM_Z - L.SLOT_DEPTH
    out = [
        bx(L.CX - 0.13, L.CX + 0.13, L.CY - 0.10, L.CY + 0.10, 0.0, L.STAND_H, st),  # the steel stand
        bx(x0, x1, y0, y1, L.STAND_H, L.SEAT_Z, c),  # the case's body: its floor is the seat face
        bx(x0, px0, y0, y1, L.SEAT_Z, L.RIM_Z, c),  # front wall
        bx(px0, px1, y0, py0, L.SEAT_Z, L.RIM_Z, c),  # right wall
        bx(px0, px1, py1, y1, L.SEAT_Z, L.RIM_Z, c),  # left wall
        bx(px1, x1, y0, y1, L.SEAT_Z, slot_z0, c),  # rear wall, under its slots
    ]
    rear_c = ((px1 + x1) / 2, L.CY, (slot_z0 + L.RIM_Z) / 2)
    holes = [(L.TAB_X - rear_c[0], ty - L.CY, L.SLOT_X, L.SLOT_Y) for ty in L.TAB_Y]
    out += list(mk.plate_with_holes(rear_c, (x1 - px1, L.CASE_LY), holes, L.RIM_Z - slot_z0, c))  # rear wall's slots
    for cy in L.CLIP_Y:  # each clip's bracket: a ledge on the front wall and two cheeks round the pivot pin
        out.append(bx(0.138, 0.210, cy - L.BRACKET_Y_HALF, cy + L.BRACKET_Y_HALF, 0.060, 0.066, st))
        for s in (-1, 1):
            a, b = sorted((cy + s * 0.0085, cy + s * 0.0145))
            out.append(bx(0.141, 0.149, a, b, 0.066, 0.102, st))
    return out


def clip_shapes(cy):
    """The clip's lever as drawn closed: a bar from its thumb tab (x 105 mm) over the pivot (x 145 mm) to its hook end
    (x 206 mm), lying 1 mm over the cover's catch lug; a wider thumb plate at the tab end."""
    z0, z1 = L.BAR_Z0, L.BAR_Z0 + L.CLIP_BAR_T
    h = L.CLIP_BAR_W / 2
    return [bx(L.CLIP_TAB_X, L.CLIP_BAR_END, cy - h, cy + h, z0, z1, L.CLIP_STEEL),
            bx(L.CLIP_TAB_PLATE[0], L.CLIP_TAB_PLATE[1], cy - L.CLIP_TAB_PLATE[2] / 2, cy + L.CLIP_TAB_PLATE[2] / 2,
               z0, z1, L.CLIP_STEEL)]


CLIPS = [mk.hinge(name, "-Y", (L.CLIP_PIV_X, cy, L.CLIP_PIV_Z), clip_shapes(cy), lower=L.CLIP_Q_LOWER,
                  upper=L.CLIP_Q_OPEN, mass=L.CLIP_MASS, spring=L.CLIP_SPRING, damping=0.02)
         for name, cy in zip(L.CLIP_NAMES, L.CLIP_Y)]
AIRBOX_PART = mk.assembly(AIRBOX, origin=(0.0, 0.0, L.TOP), shapes=airbox_shapes(), joints=CLIPS, color=L.CASE_GREY,
                          friction=L.FRICTION, tight_fit=True)


def cover_shapes():
    c = L.COVER_PP
    x0, x1, y0, y1, w = L.COVER_X0, L.COVER_X1, L.COVER_Y0, L.COVER_Y1, L.COVER_WALL
    z0, zr = L.COVER_SKIRT_Z, L.COVER_ROOF_TOP
    out = [
        bx(x0, x1, y0, y1, zr - w, zr, c),  # roof
        bx(x0, x0 + w, y0, y1, z0, zr - w, c),  # front skirt
        bx(x1 - w, x1, y0, y1, z0, zr - w, c),  # rear skirt
        bx(x0 + w, x1 - w, y0, y0 + w, z0, zr - w, c),  # right skirt
        bx(x0 + w, x1 - w, y1 - w, y1, z0, zr - w, c),  # left skirt
        bx(L.LUG_X0, L.LUG_X1, L.LUG_Y0, L.LUG_Y1, zr, L.LUG_TOP, c),  # the lift lug on the roof's front edge
        mk.cylinder(((L.SPIGOT_X0 + L.SPIGOT_X1) / 2, L.SPIGOT_Y, L.SPIGOT_Z), L.SPIGOT_R, L.SPIGOT_X1 - L.SPIGOT_X0,
                    axis="X", color=c),  # the duct's outlet
    ]
    for ty in L.TAB_Y:  # a tab: the foot of an inner rib, 12 mm below the rim, into the slot
        out.append(bx(L.TAB_X - L.TAB_T / 2, L.TAB_X + L.TAB_T / 2, ty - L.TAB_W / 2, ty + L.TAB_W / 2,
                      z0 - L.TAB_DROP, zr - w, c))
    for cy in L.CLIP_Y:  # each clip's catch lug on the front skirt
        out.append(bx(L.CLIP_LUG_X0, L.CLIP_LUG_X1, cy - L.CLIP_LUG_W / 2, cy + L.CLIP_LUG_W / 2,
                      L.CLIP_LUG_Z0, L.CLIP_LUG_Z1, c))
    return out


_STUB_C = tuple((a + b) / 2 for a, b in L.STUB_BOX)
_TINY = 0.004
DX = mk.slider("dx", "X", [mk.box(_STUB_C, (_TINY,) * 3)], lower=L.DUCT_LIM_X[0], upper=L.DUCT_LIM_X[1], mass=0.01,
               spring=L.DUCT_K, damping=L.DUCT_DAMP)
DY = mk.slider("dy", "Y", [mk.box(_STUB_C, (_TINY,) * 3)], lower=L.DUCT_LIM_Y[0], upper=L.DUCT_LIM_Y[1], mass=0.01,
               spring=L.DUCT_K, damping=L.DUCT_DAMP, parent="dx")
DZ = mk.slider("dz", "Z", cover_shapes(), lower=L.DUCT_LIM_Z[0], upper=L.DUCT_LIM_Z[1], mass=L.COVER_MASS,
               spring=L.DUCT_K, damping=L.DUCT_DAMP, parent="dy", anchor=(L.CX, L.CY, L.COVER_ROOF_TOP))
COVER_PART = mk.assembly(
    COVER, origin=(0.0, 0.0, L.TOP), color=L.COVER_PP, friction=L.FRICTION, tight_fit=True, joints=[DX, DY, DZ],
    shapes=[bx(*L.STUB_BOX[0], *L.STUB_BOX[1], *L.STUB_BOX[2], L.STAND_GREY)])  # the throttle body's stub


def filter_shapes(pleats, frame):
    """The panel filter in its own frame (origin at the bottom's centre, long side along x): a pleat pack 26 mm tall
    inside a 20 mm wide urethane frame 49 mm tall, a plain rectangle."""
    hx, hy, w = L.F_LX / 2, L.F_LY / 2, L.F_WALL
    out = [mk.box((0.0, 0.0, L.F_PLEAT_H / 2), (L.F_LX - 2 * w, L.F_LY - 2 * w, L.F_PLEAT_H), pleats)]
    h = L.F_H
    out += [bx(-hx, -hx + w, -hy, hy, 0.0, h, frame), bx(hx - w, hx, -hy, hy, 0.0, h, frame),
            bx(-hx + w, hx - w, -hy, -hy + w, 0.0, h, frame), bx(-hx + w, hx - w, hy - w, hy, 0.0, h, frame)]
    return out


OLD_PART = mk.assembly(OLD, origin=(L.OLD_START[0], L.OLD_START[1], L.TOP + L.OLD_START[2]), mass=L.F_MASS,
                       shapes=filter_shapes(L.F_PLEATS_OLD, L.F_OLD), color=L.F_OLD, friction=L.FILTER_FRICTION,
                       tight_fit=True)
NEW_PART = mk.assembly(NEW, origin=(L.NEW_START[0], L.NEW_START[1], L.TOP + L.NEW_START[2]), mass=L.F_MASS,
                       yaw=L.NEW_START_YAW, shapes=filter_shapes(L.F_PLEATS_NEW, L.F_NEW), color=L.F_NEW,
                       friction=L.FILTER_FRICTION, tight_fit=True)

MAT_N = Box("mat_new", pos=(L.MAT_NEW[0], L.MAT_NEW[1], L.TOP + L.MAT_T / 2), size=(*L.MAT_NEW_SIZE, L.MAT_T),
            mass=None, color=L.MAT_COLOR)
MAT_O = Box("mat_old", pos=(L.MAT_OLD[0], L.MAT_OLD[1], L.TOP + L.MAT_T / 2), size=(*L.MAT_OLD_SIZE, L.MAT_T),
            mass=None, color=L.MAT_COLOR)
BATTERY = Box("battery", pos=(L.BATTERY[0][0], L.BATTERY[0][1], L.TOP + L.BATTERY[2] / 2),
              size=(L.BATTERY[1][0], L.BATTERY[1][1], L.BATTERY[2]), mass=None, color=L.BATTERY_COLOR)
LIP = Box("core_support", pos=(L.LIP[0][0], L.LIP[0][1], L.TOP + L.LIP[2] / 2), size=(L.LIP[1][0], L.LIP[1][1], L.LIP[2]),
          mass=None, color=L.LIP_COLOR)
OBJECTS = [AIRBOX_PART, COVER_PART, OLD_PART, NEW_PART, MAT_N, MAT_O, BATTERY, LIP]
SPECS = {AIRBOX: AIRBOX_PART, COVER: COVER_PART, OLD: OLD_PART, NEW: NEW_PART}


# -- the one declared rule: the clips' over-centre detent -------------------------------------------------------------
@mk.on_step
def _clip_detent(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics): each clip lever is
    driven to the nearer of its two notches (closed, released): a spring toward it, so a lever pressed past the middle
    (0.675 rad) snaps over to the other notch and stays there, as an over-centre clip does."""
    box = states.get(AIRBOX)
    if box is None:
        return
    for name in L.CLIP_NAMES:
        try:
            q = float(box.joints[name])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(q):
            continue
        mk.set_targets(AIRBOX_PART, name, mk.detent_target(q, (0.0, L.CLIP_Q_OPEN), L.CLIP_Q_OPEN * 0.52))


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it knows nothing of the clips, the
# duct's sliders or a filter's seat. Offline only (never in the simulator), this model gives the scorer: (1) a closed
# block of fingers on a clip's thumb tab pushes it over centre (released), and pushing the released lever's hook end
# back over the middle snaps it closed (onto the catch lug if the cover is down); (2) a gripper closed on the cover's
# lift lug carries the cover (dz, dy, dx from its seated place), refused upward while a clip is closed, and let go the
# cover falls and the duct pulls it home; (3) a gripper closed across a filter's long frame wall carries the filter,
# turning with the gripper's heading (read from the plan, which the offline snapshot lacks), and let go it drops onto
# the seat (flat, either way round), or the rim (misseated), a mat or the table; (4) a filter cannot be pinched in the
# pocket while the cover is over it.
TAB_X_PUSH = 0.115
PAD_HALF_X = 0.0135  # a closed block of fingers at heading 0: 27 mm along x
BAR_LEN = L.CLIP_BAR_END - L.CLIP_PIV_X
GRIP_BAND = (0.019, 0.045)  # pad tips above a filter's bottom when they straddle its frame wall


def _smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def plan_yaw(plan, t: float) -> float:
    """The heading of a waypoint plan at time t (s): the downward grip's yaw, interpolated like ArmPlan does."""
    yaw0 = 0.0
    for w in plan:
        if t < w.duration:
            return yaw0 + (w.yaw - yaw0) * _smooth(t / w.duration)
        t -= w.duration
        yaw0 = w.yaw
    return yaw0


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class OfflineModel:
    def __init__(self) -> None:
        self.plans = None
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.q = [0.0, 0.0]
        self.duct = [0.0, 0.0, 0.0]
        self.f = {OLD: [L.OLD_START[0], L.OLD_START[1], L.OLD_START[2], 0.0],
                  NEW: [L.NEW_START[0], L.NEW_START[1], L.NEW_START[2], L.NEW_START_YAW]}
        self.held: dict = {}  # filter -> (arm, local grasp offset (2,), tcp z - bottom, heading at the grasp)
        self.cover_held = None  # (arm, heading, offset)
        self.last_press = [False, False]

    def _headings(self, t: float) -> dict:
        if self.plans is None:
            self.plans = demo()
        return {side: plan_yaw(self.plans[side], t) if side in self.plans else 0.0 for side in ("left", "right")}

    # -- clips ---
    def _rest_q(self) -> float:
        """A clip let go below the middle swings shut onto the catch lug under it (the cover down: 1 mm over the
        lug's top is its closed place), rests on a lifted lug, or drops to its stop with no lug under it."""
        dx, dy, dz = self.duct
        if abs(dy) > 0.012 or abs(dx) > 0.02:
            return L.CLIP_Q_LOWER
        if dz <= 0.001:
            return -0.016
        return min((dz + 0.001) / BAR_LEN, L.CLIP_Q_OPEN / 2 - 0.01)

    def _clips(self, read) -> None:
        for i, cy in enumerate(L.CLIP_Y):
            q = self.q[i]
            pressed = False
            for tcp, tips, opening in read:
                tz = float(tips[2]) - L.TOP
                if opening > 0.15 or abs(float(tcp[1]) - cy) > 0.016:
                    continue
                tx = float(tcp[0])
                tab_z = L.CLIP_PIV_Z + (TAB_X_PUSH - L.CLIP_PIV_X) * math.sin(q) + 0.003 * math.cos(q)
                if q < L.CLIP_Q_OPEN / 2 and abs(tx - TAB_X_PUSH) <= 0.012 and tz < tab_z:  # pressing the thumb tab
                    q_push = math.asin(min(1.0, max(0.0, (L.CLIP_PIV_Z + 0.003 - tz) / (L.CLIP_PIV_X - TAB_X_PUSH))))
                    q = max(q, q_push)
                    pressed = True
                elif q > L.CLIP_Q_OPEN / 2:  # a push on the released lever's upper part, from the front
                    face = tx + PAD_HALF_X
                    if tz > L.CLIP_PIV_Z + BAR_LEN * math.sin(q) or tz + 0.038 < L.CLIP_PIV_Z + 0.01 or tx > L.CLIP_PIV_X + 0.07:
                        continue
                    zlow = max(tz, L.CLIP_PIV_Z + 0.01)  # the bar must stay ahead of the block's lowest edge
                    x_bar = L.CLIP_PIV_X + (zlow - L.CLIP_PIV_Z) / max(math.tan(q), 1e-6)
                    if face >= x_bar:
                        q = min(q, math.atan2(zlow - L.CLIP_PIV_Z, max(face - L.CLIP_PIV_X, 1e-6)))
                        pressed = True
            if q > L.CLIP_Q_OPEN / 2 and not pressed:
                q = L.CLIP_Q_OPEN
            elif q > L.CLIP_Q_OPEN / 2 and pressed:
                q = q  # being pushed from the released notch: still over the middle
            elif not pressed:
                q = self._rest_q()
            self.q[i] = q

    # -- cover ---
    def _cover(self, read, head) -> None:
        dx, dy, dz = self.duct
        lug = np.array([(L.LUG_X0 + L.LUG_X1) / 2 + dx, (L.LUG_Y0 + L.LUG_Y1) / 2 + dy])
        tcp_z0 = L.LUG_TOP - 0.016 + L.PAD_TIP  # the grip height as seated
        held = None
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            tz = float(tips[2]) - L.TOP
            along = abs(float(tcp[0]) - lug[0]) <= 0.012 and abs(float(tcp[1]) - lug[1]) <= 0.015
            band = L.COVER_ROOF_TOP + dz - 0.012 <= tz <= L.LUG_TOP + dz
            closing_x = abs(math.sin(head[side])) > 0.9  # the fingers close along x
            if 0.04 <= opening <= 0.3 and along and band and closing_x:
                held = (side, tcp)
        if held is not None:
            side, tcp = held
            ndz = float(tcp[2]) - L.TOP - tcp_z0
            ndx = float(tcp[0]) - (L.LUG_X0 + L.LUG_X1) / 2
            ndy = float(tcp[1]) - (L.LUG_Y0 + L.LUG_Y1) / 2
            if any(q <= L.CLIP_Q_CLOSED_MAX for q in self.q):  # a closed clip's hook holds the cover down
                ndz = min(ndz, 0.0)
            self.duct = [min(max(ndx, L.DUCT_LIM_X[0]), L.DUCT_LIM_X[1]), min(max(ndy, L.DUCT_LIM_Y[0]), L.DUCT_LIM_Y[1]),
                         min(max(ndz, L.DUCT_LIM_Z[0]), L.DUCT_LIM_Z[1])]
        else:
            self.duct = [dx * 0.8, dy * 0.8, max(0.0, dz - 0.02)]
            if abs(self.duct[0]) < 1e-4:
                self.duct[0] = 0.0
            if abs(self.duct[1]) < 1e-4:
                self.duct[1] = 0.0
        self.cover_held = held

    # -- filters ---
    def _support(self, name: str) -> float:
        x, y, z, yaw = self.f[name]
        aligned = abs(S.wrap(2.0 * yaw)) / 2.0 <= math.radians(3.0)  # a quarter turn off does not fit the pocket
        if abs(x - L.CX) <= L.SEAT_XY_TOL and abs(y - L.CY) <= L.SEAT_XY_TOL and aligned:
            return L.SEAT_Z
        poly = np.array([(x, y)]) + np.array(L.outline_corners()) @ _rot2(yaw).T
        if S._rect_overlap(poly, L.CASE_X0, L.CASE_X1, L.CASE_Y0, L.CASE_Y1):
            return L.RIM_Z  # an edge on the rim: the filter rides up there
        for (mx, my), (sx, sy) in ((L.MAT_NEW, L.MAT_NEW_SIZE), (L.MAT_OLD, L.MAT_OLD_SIZE)):
            if abs(x - mx) <= sx / 2 and abs(y - my) <= sy / 2:
                return L.MAT_T
        return 0.0

    def _filters(self, read, head) -> None:
        for name in (OLD, NEW):
            x, y, z, yaw = self.f[name]
            held = self.held.get(name)
            if held is not None:
                side, u0, dz0, h0, yaw0 = held
                tcp, tips, opening = read[rigstate.ARMS.index(side)]
                if opening > 0.3 or opening < 0.04:
                    self.held.pop(name)
                    held = None
                else:
                    yaw = S.wrap(yaw0 + (head[side] - h0))
                    xy = np.array([float(tcp[0]), float(tcp[1])]) - _rot2(yaw) @ u0
                    self.f[name] = [float(xy[0]), float(xy[1]), float(tcp[2]) - L.TOP - dz0, yaw]
                    continue
            if held is None:
                for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
                    if not 0.04 <= opening <= 0.3:
                        continue
                    u = _rot2(-yaw) @ (np.array([float(tcp[0]), float(tcp[1])]) - np.array([x, y]))
                    wall = abs(abs(u[1]) - (L.F_LY / 2 - L.F_WALL / 2)) <= 0.012 and abs(u[0]) <= 0.09
                    normal = _rot2(yaw) @ np.array([0.0, 1.0])
                    closing = np.array([-math.sin(head[side]), math.cos(head[side])])
                    aligned = abs(float(normal @ closing)) >= 0.95
                    tz = float(tips[2]) - L.TOP - z
                    band = GRIP_BAND[0] <= tz <= GRIP_BAND[1]
                    over_cover = name == OLD and self.duct[1] < 0.15 and abs(x - L.CX) < 0.15 and z < 0.06 and \
                        abs(self.duct[2]) >= 0.0 and self._cover_over(x, y)
                    if wall and aligned and band and not over_cover:
                        self.held[name] = (side, u, float(tcp[2]) - L.TOP - z, head[side], yaw)
                        break
            if name not in self.held:
                z_support = self._support(name)
                self.f[name] = [x, y, max(z_support, z - 0.03) if z > z_support else z_support, yaw]

    def _cover_over(self, x: float, y: float) -> bool:
        """Whether the cover's footprint is over a point in the pocket."""
        dy = self.duct[1]
        return L.COVER_Y0 + dy <= y <= L.COVER_Y1 + dy and self.duct[2] < 0.12

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        self.n += 1
        if read is None:
            return
        head = self._headings(self.n / L.CONTROL_HZ)
        self._cover(read, head)
        self._clips(read)
        self._filters(read, head)

    def states(self) -> dict:
        ident = (0.0, 0.0, 0.0, 1.0)
        box = mk.state_from_poses(AIRBOX_PART, (0.0, 0.0, L.TOP, *ident),
                                  joints={n: q for n, q in zip(L.CLIP_NAMES, self.q)}, source="offline")
        cov = mk.state_from_poses(COVER_PART, (0.0, 0.0, L.TOP, *ident),
                                  joints=dict(zip(L.COVER_JOINTS, self.duct)), source="offline")
        out = {AIRBOX: box, COVER: cov}
        for name, spec in ((OLD, OLD_PART), (NEW, NEW_PART)):
            x, y, z, yaw = self.f[name]
            q = mk.quat_from_ypr(yaw, 0.0, 0.0)
            out[name] = mk.state_from_poses(spec, (x, y, L.TOP + z, *q), source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(OLD), "source", "sim") != "sim"


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
    """With TASK_TRACE=<file>, append one JSON line per control step: every key part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Old filter out on a mat, new one flat in the seat, cover down with both tabs in
    their slots, both clips closed, both grippers 30 mm clear and every part still for 1 s, no latched zero; held 1 s.
    This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline model)."""
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
    row["source"] = getattr(parts.get(OLD), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the duct was pulled off its stub or a filter fell off the table (latched), or unless every part is let go
    and still; 1.0 old out, new seated, cover seated, both clips closed; 0.5 the same with a clip open; 0.25 old out and new seated, cover not seated; 0 otherwise. The
    score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
OPEN = L.FINGER_OPEN  # 94 mm
CLOSED = 0.0
GRIP_LUG = -(L.LUG_X1 - L.LUG_X0 - 0.008) / 2  # 12 mm commanded across the 20 mm lug
GRIP_FRAME = -(L.F_WALL - 0.008) / 2  # 12 mm across the 20 mm frame
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # the left arm closes along x at -90, the right at +90

TAB_TCP_HOVER, TAB_TCP_TOUCH, TAB_TCP_PRESS = 0.128, 0.115, 0.082  # TCP heights over a clip's thumb tab
TAB_X = TAB_X_PUSH
PUSH_X0, PUSH_X1, PUSH_Z = 0.128, 0.185, 0.138  # the block of fingers that shuts a released clip: pushes the hook
PUSH_HOVER = 0.160  # end toward the cover from the front
LUG_CX = (L.LUG_X0 + L.LUG_X1) / 2
LUG_CY = (L.LUG_Y0 + L.LUG_Y1) / 2
LUG_TCP = L.LUG_TOP - 0.016 + L.PAD_TIP  # pad tips 16 mm under the lug's top, 4 mm over the roof
LIFT = 0.040  # commanded; the arm sags 4 to 8 mm under the cover's weight and the duct's pull
SHIFT = 0.240  # the cover is slid 240 mm toward +y: off the pocket
HOVER_LUG = 0.170
FRAME_TCP_DZ = L.F_H  # the TCP is a filter's height above its bottom when the pad tips are 13 mm under its top
FILTER_LIFT = 0.045  # a filter is lifted 45 mm: its bottom 15 mm over the rim
WAIT_R = (0.200, -0.300, 0.165)
STAGE_L = (0.230, 0.360, 0.170)
PARK_L = (0.220, 0.400, 0.170)
PARK_R = WAIT_R

# grasp points
WALL_Y = L.CY - L.F_LY / 2 + L.F_WALL / 2  # the old filter's right frame wall's centre line, y -0.0665
G_OLD = (L.CX + L.GRASP_DX, WALL_Y)  # TCP x, y when grasping the filter in the seat
# the new filter lies turned -90 degrees: its front long wall (x) is the one that becomes the right wall in the seat
G_NEW = (L.NEW_START[0] - L.F_LY / 2 + L.F_WALL / 2, L.NEW_START[1] - L.GRASP_DX)
# put-down on the old mat: the filter's centre at the mat's centre
G_MAT_OLD = (L.MAT_OLD[0] + L.GRASP_DX, L.MAT_OLD[1] - L.F_LY / 2 + L.F_WALL / 2)


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def tab_moves(y, yaw, first=True):
    """The right arm shuts a clip's thumb tab down: closed fingers over the tab, down onto it, push it past the middle
    (the clip releases), up."""
    return [
        _w(2.0 if first else 1.5, TAB_X, y, TAB_TCP_HOVER, CLOSED, "closed fingers over the clip's thumb tab", yaw),
        _w(0.6, TAB_X, y, TAB_TCP_TOUCH, CLOSED, "down onto the tab", yaw),
        _w(1.2, TAB_X, y, TAB_TCP_PRESS, CLOSED, "press it down past the middle: the clip releases", yaw),
        _w(0.4, TAB_X, y, TAB_TCP_PRESS, CLOSED, "settle", yaw),
        _w(0.8, TAB_X, y, 0.150, CLOSED, "up", yaw),
    ]


def push_moves(y, yaw, first=True):
    """The right arm shuts a released clip: closed fingers in front of the raised lever, push its upper end back over
    the middle, hold while it drops onto the catch lug, back off."""
    return [
        _w(2.0 if first else 1.2, PUSH_X0, y, PUSH_HOVER, CLOSED, "closed fingers in front of the released clip", yaw),
        _w(0.8, PUSH_X0, y, PUSH_Z, CLOSED, "down beside the lever", yaw),
        _w(1.5, PUSH_X1, y, PUSH_Z, CLOSED, "push the lever's upper end back over the middle", yaw),
        _w(0.8, PUSH_X1, y, PUSH_Z, CLOSED, "hold while it drops onto the lug", yaw),
        _w(0.8, PUSH_X0, y, PUSH_Z, CLOSED, "back off", yaw),
        _w(0.8, PUSH_X0, y, PUSH_HOVER, CLOSED, "up", yaw),
    ]


def lug_grip():
    x, y = LUG_CX, LUG_CY
    return [
        _w(1.8, x, y, HOVER_LUG, OPEN, "over the cover's lift lug", YAW_L),
        _w(0.8, x, y, LUG_TCP + 0.025, OPEN, "down to just above the roof", YAW_L),
        _w(1.2, x, y, LUG_TCP, OPEN, "slowly down round the lug", YAW_L),
        _w(0.4, x, y, LUG_TCP, OPEN, "settle", YAW_L),
        _w(1.5, x, y, LUG_TCP, GRIP_LUG, "grip the lug across its 20 mm", YAW_L),
        _w(0.4, x, y, LUG_TCP, GRIP_LUG, "settle the grip", YAW_L),
    ]


def cover_aside():
    x, y = LUG_CX, LUG_CY
    return [
        _w(1.8, x, y, LUG_TCP + LIFT, GRIP_LUG, "lift the cover 40 mm (commanded)", YAW_L),
        _w(3.2, x, y + SHIFT, LUG_TCP + LIFT, GRIP_LUG, "slide it 240 mm toward the duct, off the pocket", YAW_L),
    ]


def cover_back():
    x, y = LUG_CX, LUG_CY
    return [
        _w(3.2, x, y, LUG_TCP + LIFT, GRIP_LUG, "slide the cover back over the pocket", YAW_L),
        _w(1.5, x, y, LUG_TCP - 0.002, GRIP_LUG, "lower it: tabs into the slots", YAW_L),
        _w(0.5, x, y, LUG_TCP - 0.002, GRIP_LUG, "settle", YAW_L),
        _w(1.5, x, y, LUG_TCP - 0.002, OPEN, "let go of the lug", YAW_L),
        _w(1.0, x, y, HOVER_LUG, OPEN, "straight up", YAW_L),
    ]


def filter_out():
    """The right arm: pinch the old filter's right frame wall 30 mm toward its front, lift it 45 mm, carry it over the
    new filter's mat to the old mat, set it down."""
    gx, gy = G_OLD
    hover = 0.150
    z_grip = FRAME_TCP_DZ + L.SEAT_Z
    mx, my = G_MAT_OLD
    z_mat = FRAME_TCP_DZ + L.MAT_T
    z_carry = FRAME_TCP_DZ + L.SEAT_Z + FILTER_LIFT
    return [
        _w(2.0, gx, gy, hover, OPEN, "over the old filter's right frame wall", 0.0),
        _w(0.8, gx, gy, z_grip + 0.020, OPEN, "down to just above it", 0.0),
        _w(1.2, gx, gy, z_grip, OPEN, "slowly down round the 20 mm frame", 0.0),
        _w(0.4, gx, gy, z_grip, OPEN, "settle", 0.0),
        _w(1.5, gx, gy, z_grip, GRIP_FRAME, "pinch the frame", 0.0),
        _w(0.4, gx, gy, z_grip, GRIP_FRAME, "settle the grip", 0.0),
        _w(1.5, gx, gy, z_carry, GRIP_FRAME, "lift the old filter 45 mm out of the pocket", 0.0),
        _w(3.5, mx, my, z_carry, GRIP_FRAME, "carry it over the new filter to the old mat", 0.0),
        _w(1.6, mx, my, z_mat, GRIP_FRAME, "lower it onto the mat", 0.0),
        _w(0.4, mx, my, z_mat, GRIP_FRAME, "settle", 0.0),
        _w(1.5, mx, my, z_mat, OPEN, "let go of the old filter", 0.0),
        _w(0.8, mx, my, 0.120, OPEN, "straight up (kept low: the old mat is near the reach edge)", 0.0),
    ]


def filter_in():
    """The right arm: pinch the new filter's front frame wall (it lies turned a quarter turn), lift it, carry it over
    the pocket turning it a quarter turn so its long side lies along the pocket, lower it onto the seat, let go."""
    nx, ny = G_NEW
    z_grip = FRAME_TCP_DZ + L.MAT_T
    z_carry = z_grip + FILTER_LIFT + 0.043
    gx, gy = G_OLD
    z_seat = FRAME_TCP_DZ + L.SEAT_Z
    z_in = z_seat - 0.001
    return [
        _w(2.2, nx, ny, 0.140, OPEN, "over the new filter's front frame wall", -math.pi / 2),
        _w(0.8, nx, ny, z_grip + 0.020, OPEN, "down to just above it", -math.pi / 2),
        _w(1.2, nx, ny, z_grip, OPEN, "slowly down round the 20 mm frame", -math.pi / 2),
        _w(0.4, nx, ny, z_grip, OPEN, "settle", -math.pi / 2),
        _w(1.5, nx, ny, z_grip, GRIP_FRAME, "pinch the frame", -math.pi / 2),
        _w(0.4, nx, ny, z_grip, GRIP_FRAME, "settle the grip", -math.pi / 2),
        _w(1.5, nx, ny, z_grip + FILTER_LIFT + 0.043 - 0.0, GRIP_FRAME, "lift the new filter", -math.pi / 2),
        _w(3.5, gx, gy, z_carry, GRIP_FRAME, "carry it over the pocket, turning it a quarter turn", 0.0),
        _w(1.2, gx, gy, z_seat + 0.020, GRIP_FRAME, "down to just above the seat", 0.0),
        _w(1.2, gx, gy, z_in, GRIP_FRAME, "slowly down onto the seat", 0.0),
        _w(0.4, gx, gy, z_in, GRIP_FRAME, "settle", 0.0),
        _w(1.5, gx, gy, z_in, OPEN, "let go of the new filter", 0.0),
        _w(0.8, gx, gy, 0.150, OPEN, "straight up", 0.0),
    ]


VARIANTS = ("one_clip",)  # counterfactuals for the README: the left clip is never shut (0.5)


def _hold(plan, until, x, y, z, finger, yaw, label):
    """Append a wait at (x, y, z) so the plan lasts ``until`` seconds (nothing if it already does)."""
    gap = until - _duration(plan)
    if gap > 1e-6:
        plan.append(_w(gap, x, y, z, finger, label, yaw))


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work (one arm near the front of the airbox at a time: two forearms at the clips overlap): the right arm
    presses both clips' thumb tabs (the clips release); the left arm, waiting aside, grips the cover's lift lug, lifts
    the cover 40 mm and slides it 240 mm toward the duct; the right arm takes the old filter out onto its mat and
    brings the new one in, turned a quarter turn, then backs away; the left arm slides the cover back, lowers it, lets
    go and parks; the right arm shuts both clips."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = tab_moves(L.CLIP_Y[0], 0.0) + tab_moves(L.CLIP_Y[1], 0.0, first=False)
    t_tabs = _duration(right)
    right.append(_w(2.0, *WAIT_R, OPEN, "aside, waiting for the cover", 0.0))
    left = [_w(2.0, *STAGE_L, OPEN, "aside, waiting for the clips", 0.0)]
    _hold(left, t_tabs + 1.0, *STAGE_L, OPEN, 0.0, "wait for the clips to be released")
    left += lug_grip() + cover_aside()
    t_aside = _duration(left)
    _hold(right, t_aside, *WAIT_R, OPEN, 0.0, "wait for the cover to be aside")
    right += filter_out() + filter_in()
    t_up = _duration(right) - 0.8  # the new filter is let go: the right arm starts to rise (0.8 s)
    _hold(left, t_up, LUG_CX, LUG_CY + SHIFT, LUG_TCP + LIFT, GRIP_LUG, YAW_L, "hold the cover aside while the filters change")
    left += cover_back()
    t_down = t_up + 3.2 + 1.5 + 0.5  # the cover is lowered and settled
    left.append(_w(2.0, *PARK_L, OPEN, "aside", 0.0))
    right.append(_w(2.0, *WAIT_R, OPEN, "back away", 0.0))
    _hold(right, t_down + 4.0, *WAIT_R, OPEN, 0.0, "wait for the left arm to be clear")
    right += push_moves(L.CLIP_Y[0], 0.0)
    if variant != "one_clip":
        right += push_moves(L.CLIP_Y[1], 0.0, first=False)
    right.append(_w(1.5, *PARK_R, OPEN, "aside", 0.0))
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      one_clip  the left clip is never shut: 0.5"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
