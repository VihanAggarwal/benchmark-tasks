"""On the station table, real size: a 4 fl oz (118 mL) amber HDPE children's liquid-medicine bottle, cap off, 118 mL of
syrup in it (49 mm round body, 105 mm tall to the lip, 22 mm opening, 170 g), standing at x 30.0 cm, y -18.0 cm (its
cap put away, not on the table); a 30 mL round dosing cup (40 mm across, 38 mm tall, 8 g, 2 mm walls; clear,
graduated 5 to 30 mL with numerals 10, 20 and 30, a red 10 mL line) standing on the white 100 mm platform of a pocket
scale (14 mm high, fixed) at x 30.0 cm, y 0.

Goal: 10 mL of syrup in the cup (the 10 mL line, 9.0 to 11.0 mL), poured with the cup held up and level, none
spilled, the cup set back on the scale and the bottle put back upright, hands off.

The syrup is a volume ledger (scoring.py): the surface of the 118 mL in the bottle follows the bottle's real pose, syrup
goes over the lip at a laminar film rate capped at 3 mL/s when the surface stands above the lip's lowest point, and
falls into the cup if that point is over its opening. The bottle and the cup are mechkit compounds
(mechkit.py, next to this file); the arms are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Trace: TASK_TRACE=<file> (one JSON line per
control step).  Calibration: DEMO_POUR_HOLD=<seconds> sets the time the bottle stays at full tilt (default 2.2).
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
import pourplan as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (tilted grips for the demo: wraps yam_arena.demo.grip_rotation)
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Pour 10 mL of syrup from the bottle into the dosing cup, holding the cup up and level while you pour. "
    "Then set the cup back on the scale and the bottle back where it stood."
)

# -- the parts -------------------------------------------------------------------------------------------------------
SCALE = Box("scale_pan", pos=(L.PAN_X, L.PAN_Y, L.TOP + L.PAN_SIZE[2] / 2), size=L.PAN_SIZE, color=L.PAN_WHITE)

CUP_NAME, BOTTLE_NAME = "dosing_cup", "syrup_bottle"


def panel_angle(k: int) -> float:
    """The direction (rad, about z from +x) that wall panel ``k`` of the round cup faces; panel 0 faces +x."""
    return 2.0 * math.pi * (k % L.CUP_SIDES) / L.CUP_SIDES


def _ring(k: int, radius: float, z: float, size, color=None, hidden: bool = False, t: float = 0.0):
    """A box on the cup's ring: its centre ``radius`` out from the axis toward panel ``k`` (and ``t`` along the panel's
    +tangential axis, (-sin phi, cos phi)) at height ``z``, its size (radial, along the panel, height) turned to face
    that way."""
    phi = panel_angle(k)
    x = radius * math.cos(phi) - t * math.sin(phi)
    y = radius * math.sin(phi) + t * math.cos(phi)
    return mk.box((x, y, z), size, color, ypr=(phi, 0.0, 0.0), hidden=hidden)


def mark_height(ml: float) -> float:
    """How tall the graduation at ``ml`` is drawn: 6 mm for the 10 mL line, 4 mm for a ring, 1.5 mm for a tick."""
    if ml == L.DOSE_ML:
        return L.LINE_W
    return L.RING_W if ml in L.RING_ML else L.TICK_W


def _mark(ml: float, z: float, k: int, inside: bool):
    """One graduation strip at ``ml`` (height ``z`` over the cup's base) on panel ``k``, drawn bolder than real print
    so the cameras resolve it: the 10 mL line red and 6 mm tall, the 20 and 30 mL rings near black and 4 mm tall, the
    5, 15 and 25 mL ticks near black and 1.5 mm tall, each MARK_L along its panel; outside 0.6 mm proud of the panel,
    inside 0.3 mm proud, each sunk 0.3 mm into it (its back face inside the panel, not on its face)."""
    line = ml == L.DOSE_ML
    color = L.LINE_RED if line else L.INK
    height = mark_height(ml)
    proud = L.MARK_IN_T if inside else L.LINE_T
    face = L.CUP_INNER_HALF if inside else L.CUP_OUTER / 2
    r = face + (L.MARK_EMBED - proud) / 2 if inside else face + (proud - L.MARK_EMBED) / 2
    return _ring(k, r, z, (proud + L.MARK_EMBED, L.MARK_L, height), color)


def _shell():
    """The cup: a round floor (40 mm across, 2 mm, opaque pale clear-plastic blue) and a ring of CUP_SIDES flat wall
    panels, 2 mm thick, up to the 38 mm rim: their inside faces 18 mm from the axis (36 mm across), their outside faces
    20 mm. Each panel is two boxes of the same radial span and width: the lower part, from CUP_SEAM over the floor to
    CUP_SEAM under the lip, hidden (a collider drawn not at all: the clear wall; in CUP_WALL_FALLBACK if a renderer
    ignores the hide), and the rolled lip, the top RIM_H, drawn opaque in CLEAR_EDGE. Every face stands 0.1 mm clear
    of its neighbours (the floor, the next panel at the inner corners, the lip), so no two faces coincide."""
    z0, z1 = L.CUP_FLOOR + L.CUP_SEAM, L.CUP_H - L.RIM_H - L.CUP_SEAM
    floor = mk.cylinder((0.0, 0.0, L.CUP_FLOOR / 2), L.CUP_FLOOR_R, L.CUP_FLOOR, "Z", L.CUP_BASE)
    r = L.CUP_INNER_HALF + L.CUP_WALL / 2
    walls = tuple(_ring(k, r, (z0 + z1) / 2, (L.CUP_WALL, L.PANEL_L, z1 - z0), L.CUP_WALL_FALLBACK, hidden=True)
                  for k in range(L.CUP_SIDES))
    lips = tuple(_ring(k, r, L.CUP_H - L.RIM_H / 2, (L.CUP_WALL, L.PANEL_L, L.RIM_H), L.CLEAR_EDGE)
                 for k in range(L.CUP_SIDES))
    return (floor,) + walls + lips


def _ring_dist(k: int, j: int) -> int:
    d = (k - j) % L.CUP_SIDES
    return min(d, L.CUP_SIDES - d)


GRIPPED = (0, L.CUP_SIDES // 2)  # the panels facing +x and -x: the left hand's fingers close on them
FACING_ARMS = (L.CUP_SIDES // 4, 3 * L.CUP_SIDES // 4)  # the panels facing +y and -y
NUMERAL_PANELS = tuple(k for pair in L.NUMERAL_PANELS.values() for k in pair)  # 6, 7, 17, 18: no ring on them
# Each ring is drawn once round the cup, on one face of each panel: outside on the panels of the top camera's (-x) half
# that the hand does not grip and that carry no numeral, inside on the rest (the far half, and the three gripped near
# panels, seen through the clear wall); none on the four numeral panels, inside or out.
OUTER_RING_PANELS = tuple(k for k in range(L.CUP_SIDES) if math.cos(panel_angle(k)) < 1e-9
                          and min(_ring_dist(k, g) for g in GRIPPED) > L.OUTER_RING_SKIP and k not in NUMERAL_PANELS)
INNER_RING_PANELS = tuple(k for k in range(L.CUP_SIDES) if k not in OUTER_RING_PANELS and k not in NUMERAL_PANELS)
OUTER_TICK_PANELS = tuple(L.TICK_PANELS_POS)

# The graduations (5 to 30 mL). Rings at 10, 20 and 30 mL (the 10 mL line red), each on one face of a panel: outside on
# OUTER_RING_PANELS (6), inside on INNER_RING_PANELS (14), so no ring shows twice. Ticks at 5, 15 and 25 mL outside on
# TICK_PANELS_POS. Thin strips of the cup's own body, display only; none on the outside of a gripped panel.
# OUTER_INFO / INNER_INFO: (panel, mL) of each.
OUTER_INFO = tuple((k, ml) for ml in L.GRAD_ML
                   for k in (OUTER_RING_PANELS if ml in L.RING_ML else OUTER_TICK_PANELS))
INNER_INFO = tuple((k, ml) for ml in L.RING_ML for k in INNER_RING_PANELS)
_Z_OF = dict(zip(L.GRAD_ML, L.GRAD_Z))
SHELL = _shell()
WALLS = SHELL[1:1 + L.CUP_SIDES]  # the hidden lower parts of the 24 panels (colliders, not drawn)
LIPS = SHELL[1 + L.CUP_SIDES:]  # the 24 lips (drawn)
OUTER_MARKS = tuple(_mark(ml, _Z_OF[ml], k, False) for k, ml in OUTER_INFO)
INNER_MARKS = tuple(_mark(ml, _Z_OF[ml], k, True) for k, ml in INNER_INFO)
LINE_STRIPS = tuple(s for s, (k, ml) in zip(OUTER_MARKS, OUTER_INFO) if ml == L.DOSE_ML)  # the red 10 mL line outside, 6
INNER_LINE_STRIPS = tuple(s for s, (k, ml) in zip(INNER_MARKS, INNER_INFO) if ml == L.DOSE_ML)  # ... inside, 14

# Printed numerals: seven-segment digits, one per panel, standing on the panel's outside face (NUM_PROUD proud, sunk
# MARK_EMBED). Segment centres in the panel's frame (t along its +tangential axis, the outside reader's right; z over
# the digit's centre): a, g, d the horizontals (top, middle, bottom), b, c the right verticals (upper, lower), f, e
# the left ones.
SEGMENTS = {
    "a": (0.0, 0.003, "h"), "g": (0.0, 0.0, "h"), "d": (0.0, -0.003, "h"),
    "b": (L.NUM_SEG_T, 0.0015, "v"), "c": (L.NUM_SEG_T, -0.0015, "v"),
    "f": (-L.NUM_SEG_T, 0.0015, "v"), "e": (-L.NUM_SEG_T, -0.0015, "v"),
}
DIGIT_SEGMENTS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd"}


def _digit(d: int, k: int, zc: float, color):
    """The digit ``d`` (0 to 3) on the outside of panel ``k``, centred at height ``zc``: one box per lit segment, the
    horizontals NUM_W x NUM_STROKE, the verticals NUM_STROKE wide and NUM_SEG_V tall (0.1 mm clear of each other)."""
    r = L.CUP_OUTER / 2 + (L.NUM_PROUD - L.MARK_EMBED) / 2
    thick = L.NUM_PROUD + L.MARK_EMBED
    out = []
    for name in DIGIT_SEGMENTS[d]:
        t, dz, kind = SEGMENTS[name]
        size = (thick, L.NUM_W, L.NUM_STROKE) if kind == "h" else (thick, L.NUM_STROKE, L.NUM_SEG_V)
        out.append(_ring(k, r, zc + dz, size, color, t=t))
    return tuple(out)


# NUMERAL_INFO: (panel, value, digit, segment count) of each digit; NUMERALS: their boxes, in that order.
NUMERAL_INFO = tuple((k, v, int(ch), len(DIGIT_SEGMENTS[int(ch)]))
                     for v in (10, 20, 30) for pair in L.NUMERAL_PANELS.values() for k, ch in zip(pair, str(v)))
NUMERALS = tuple(seg for k, v, d, n in NUMERAL_INFO
                 for seg in _digit(d, k, L.NUM_Z[v], L.LINE_RED if v == L.DOSE_ML else L.INK))
# The syrup's level: the one declared display rule. A thin golden disc in the cup (opaque, seen through the clear
# wall, which is not drawn) that a slider drives, every control step, to the ledger's level (``syrup_target``). Empty, it is
# parked inside the solid floor (its top 0.05 mm under the floor's top), so an empty cup shows no syrup. The scorer
# reads poses, not the disc.
SYRUP_PLATE = mk.slider(
    "syrup", "Z",
    [mk.cylinder((0.0, 0.0, L.CUP_FLOOR - L.SYRUP_SINK - L.SYRUP_T / 2), L.SYRUP_R, L.SYRUP_T, "Z", L.SYRUP_GOLD)],
    upper=L.SYRUP_RANGE, lower=0.0, mass=L.SYRUP_MASS, spring=L.SYRUP_SPRING, damping=0.3,
)
CUP = mk.assembly(
    CUP_NAME,
    origin=(L.CUP_X, L.CUP_Y, L.CUP_Z),
    mass=L.CUP_MASS,
    shapes=SHELL + OUTER_MARKS + INNER_MARKS + NUMERALS,
    joints=[SYRUP_PLATE],
    color=L.CUP_BASE,
)
BOTTLE = mk.assembly(
    BOTTLE_NAME,
    origin=(L.BOTTLE_X, L.BOTTLE_Y, L.TOP),
    mass=L.BOTTLE_MASS,
    shapes=[
        mk.cylinder((0.0, 0.0, L.BODY_H / 2), L.BODY_R, L.BODY_H, "Z", L.AMBER),
        mk.cylinder((0.0, 0.0, (L.BODY_H + L.NECK_TOP) / 2), L.NECK_R, L.NECK_TOP - L.BODY_H, "Z", L.AMBER),
        # look: a plain white paper label round the body, 0.3 mm proud; the opening, a dark disc in the lip
        mk.cylinder((0.0, 0.0, (L.LABEL_Z0 + L.LABEL_Z1) / 2), L.BODY_R + L.LABEL_T, L.LABEL_Z1 - L.LABEL_Z0, "Z",
                    L.LABEL_WHITE),
        mk.cylinder((0.0, 0.0, L.NECK_TOP + 0.0001 - L.OPENING_T / 2), L.LIP_R, L.OPENING_T, "Z", L.OPENING_DARK),
    ],
    color=L.AMBER,
)
OBJECTS = [SCALE, CUP, BOTTLE]

_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def syrup_target(v_cup: float) -> float:
    """The disc's travel for ``v_cup`` m^3 of syrup in the cup: its top at V / area over the floor's top (from its
    parked place 0.05 mm under that), never negative, stopped just under the brim; 0 (parked inside the floor) for an
    empty cup and for anything that is not a finite volume."""
    try:
        v = float(v_cup)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0.0:
        return 0.0
    return min(v / L.CUP_AREA + L.SYRUP_SINK, L.SYRUP_RANGE)


@mk.on_step
def _syrup_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    mk.set_targets(CUP, "syrup", syrup_target(_TRACKER.v_cup))


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and keeps it upright (it never tips a held bottle), and it has no
# liquids. Offline only (never in the simulator), the bottle's pose while the right hand holds it is rebuilt from the
# hand: the pad-tip point and approach axis (rigstate) and the carried point the checker follows give two directions
# that fix the bottle's rotation since the grasp (TRIAD). The scorer then runs on that, so the ledger sees the demo's
# real tilt. The cup is carried upright by the checker's own proxy, as it is held level in the simulator too.
class OfflineBottle:
    HELD_OPENING = 0.6  # a gripper closed past this (0 closed .. 1 open) on the bottle holds it

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grasp = None  # (approach axis, carried point minus TCP) at the first held step
        self.prev = None

    def state(self, bottle, arms):
        read = rigstate.read_arms(arms)
        if read is None:
            self.grasp, self.prev = None, None
            return bottle
        tcp, tips, opening = read[rigstate.ARMS.index("right")]
        centre = np.array(bottle.pos, dtype=float)  # the checker carries the box centre with the hand (any tilt)
        held = opening < self.HELD_OPENING and float(np.linalg.norm(tips - centre)) < 0.05
        if not held:
            self.grasp, self.prev = None, None
            return bottle
        ap = tips - tcp
        ap = ap / (float(np.linalg.norm(ap)) or 1.0)
        arm = centre - tcp
        if self.grasp is None:
            self.grasp = (ap, arm)
        r = _triad(self.grasp[0], self.grasp[1], ap, arm)
        origin = centre - r @ np.array([0.0, 0.0, L.NECK_TOP / 2])
        quat = P.matrix_quat(r)
        vel = np.zeros(6) if self.prev is None else np.concatenate(
            [(origin - self.prev[0]) / 0.05, _spin(self.prev[1], r) / 0.05])
        self.prev = (origin, r)
        return mk.state_from_poses(BOTTLE, (*origin, *quat), vel, source="offline")


def _frame(a, b):
    a = a / (float(np.linalg.norm(a)) or 1.0)
    c = np.cross(a, b)
    n = float(np.linalg.norm(c))
    if n < 1e-9:  # parallel: any perpendicular will do
        c = np.cross(a, np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
        n = float(np.linalg.norm(c))
    c = c / n
    return np.column_stack([a, c, np.cross(a, c)])


def _triad(a0, b0, a1, b1) -> np.ndarray:
    """The rotation that takes direction a0 to a1 and the plane (a0, b0) to the plane (a1, b1)."""
    return _frame(a1, b1) @ _frame(a0, b0).T


def _spin(r0, r1) -> np.ndarray:
    d = r1 @ r0.T
    ang = math.acos(max(-1.0, min(1.0, (float(np.trace(d)) - 1.0) / 2.0)))
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    return axis / (2.0 * math.sin(ang)) * ang


OfflineModel = OfflineBottle  # the name the other tasks use for their offline stand-in
_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(BOTTLE_NAME), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _hands_clear(arms, bottle, cup) -> bool:
    """Both grippers' finger zones 3 cm or more from both parts; False if the arms or a distance cannot be read."""
    def dist(points):
        return np.minimum(bottle.distance(points), cup.distance(points))

    try:
        return bool(rigstate.released(arms, dist, L.LET_GO))
    except Exception:  # noqa: BLE001 - fail closed: not released
        return False


def success(objects) -> bool:
    """10 mL (9.0 to 11.0) in the cup, poured with it held up and level, 0.5 mL or less spilled, the cup on the scale,
    the bottle back upright on its mark, both grippers 3 cm clear, both parts still; held 1 s. This runs first every
    control step, so it also feeds the ledger (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    bottle, cup = objects[BOTTLE_NAME], objects[CUP_NAME]
    arms = rigstate.arms()
    if no_simulator({BOTTLE_NAME: bottle}):
        bottle = _OFFLINE.state(bottle, arms)
    clear = _hands_clear(arms, bottle, cup)
    result = _TRACKER.update(bottle, cup, clear)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "resets": _TRACKER.resets, **result.details}
    try:
        row.update(bottle_origin=bottle.origin, cup_origin=cup.origin, cup_low_z=float(cup.low[2]),
                   bottle_low_z=float(bottle.low[2]), source=getattr(bottle, "source", None))
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = "unreadable part state"
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
    """0 if more than 15 mL ever stood in the cup, the cup was tipped over 45 degrees, or over 10 mL was spilled. Else
    by the mL in the cup: 1.0 for 9.0 to 11.0 mL with 0.5 mL or less spilled, none poured with the cup down or tilted,
    the cup on the scale, the bottle put back and both grippers clear; 0.75 the same dose with any of that not met;
    0.5 for 8.0 to 12.0 mL (also the cap when the dose was poured with the cup not held up and level, or over 3 mL
    spilled); 0.25 for at least 4 mL; 0 below. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm ahead of them along the approach axis. Finger targets: the joint value is minus half the commanded gap (8 mm
# narrower than the part).
OPEN = -0.04695  # 94 mm
BOTTLE_GRIP = -(2 * L.BODY_R - 0.008) / 2  # 41 mm commanded across the 49 mm body
CUP_GRIP = -(L.CUP_OUTER - 0.008) / 2  # 32 mm commanded across the 40 mm cup (its +-x panels)
STAGE = 0.12  # the hands wait this far back along their approach axes
WITHDRAW = 0.16  # ... and the left hand withdraws this far, so that its fingers are 3 cm clear of the cup
WITHDRAW_R, WITHDRAW_UP = 0.10, 0.12  # the right hand: back and up (straight back runs into the arm's own base)
THETA = P.THETA_HOLD
T_UP, T_DOWN = 2.5, 2.5  # the tilt in and back (s)
POUR_HOLD = 2.2  # at full tilt (s); DEMO_POUR_HOLD overrides it for calibration (offline: 10.1 mL at 2.2 s, 2.9 mL/s)
CUP_SET_PRESS = 0.001  # the cup is lowered commanded this far below the pan's top

_YAW_R = tiltgrip.tilted(0.0, P.PITCH, 0.0)
_YAW_L = tiltgrip.tilted(-math.pi / 2, 0.0, -P.ROLL_L)


def _yaw_r(theta: float) -> float:
    return tiltgrip.tilted(0.0, P.PITCH, -theta)


def _wp(duration, pos, finger, label, yaw=0.0):
    return Waypoint(duration, tuple(float(v) for v in pos), finger, label, yaw)


def pour_hold() -> float:
    try:
        return float(os.environ.get("DEMO_POUR_HOLD", POUR_HOLD))
    except ValueError:
        return POUR_HOLD


def right_plan(hold: float | None = None) -> list[Waypoint]:
    """Right arm: grip the bottle at its body, lift it, bring it over the held cup, tilt it to 76 degrees about its
    grip point, hold, tilt it back, put it back on its mark and let go."""
    hold = pour_hold() if hold is None else hold
    a0 = P.grip_start()
    a_pour = P.pour_anchor(THETA)
    a_up = np.array([a0[0], a0[1], a_pour[2]])  # straight up from the start
    ap0 = P.right_approach(0.0)
    tcp0 = P.right_tcp(a0, 0.0)
    return [
        _wp(2.2, tcp0 - STAGE * ap0, OPEN, "beside the bottle", _YAW_R),
        _wp(1.5, tcp0, OPEN, "in to the bottle", _YAW_R),
        _wp(1.5, tcp0, BOTTLE_GRIP, "grip the bottle", _YAW_R),
        _wp(1.4, P.right_tcp(a_up, 0.0), BOTTLE_GRIP, "lift it straight up", _YAW_R),
        _wp(1.6, P.right_tcp(a_pour, 0.0), BOTTLE_GRIP, "over to the cup, upright", _YAW_R),
        _wp(T_UP, P.right_tcp(a_pour, THETA), BOTTLE_GRIP, "tilt to pour", _yaw_r(THETA)),
        _wp(hold, P.right_tcp(a_pour, THETA), BOTTLE_GRIP, "hold the pour", _yaw_r(THETA)),
        _wp(T_DOWN, P.right_tcp(a_pour, 0.0), BOTTLE_GRIP, "tilt back upright", _YAW_R),
        _wp(1.8, P.right_tcp(a0 + np.array([0.0, 0.0, 0.03]), 0.0), BOTTLE_GRIP, "back over its mark", _YAW_R),
        _wp(1.2, P.right_tcp(a0, 0.0) + np.array([0.0, 0.0, -0.0005]), BOTTLE_GRIP, "down on its mark", _YAW_R),
        _wp(1.5, P.right_tcp(a0, 0.0) + np.array([0.0, 0.0, -0.0005]), OPEN, "let go", _YAW_R),
        _wp(1.5, tcp0 - WITHDRAW_R * ap0 + np.array([0.0, 0.0, WITHDRAW_UP]), OPEN, "withdraw", _YAW_R),
    ]


def left_plan(hold: float | None = None) -> list[Waypoint]:
    """Left arm: grip the cup at its walls, lift it 32 mm (and 6 mm toward +y, away from the bottle) and hold it level
    while the syrup goes in, set it down on the scale and let go."""
    hold = pour_hold() if hold is None else hold
    c0 = np.array([L.CUP_X, L.CUP_Y, L.CUP_Z])
    ap = P.left_approach()
    tcp0 = P.left_tcp(c0)
    up = np.array([0.0, P.CUP_HOLD_DY, P.CUP_LIFT])
    # the right arm's plan, in time: the pour ends (the bottle upright again) at this many seconds
    t_right_upright = 2.2 + 1.5 + 1.5 + 1.4 + 1.6 + T_UP + hold + T_DOWN
    t_left_lifted = 2.2 + 1.5 + 1.5 + 1.2
    return [
        _wp(2.2, tcp0 - STAGE * ap, OPEN, "beside the cup", _YAW_L),
        _wp(1.5, tcp0, OPEN, "in to the cup", _YAW_L),
        _wp(1.5, tcp0, CUP_GRIP, "grip the cup", _YAW_L),
        _wp(1.2, tcp0 + up, CUP_GRIP, "lift it 32 mm", _YAW_L),
        _wp(t_right_upright - t_left_lifted, tcp0 + up, CUP_GRIP, "hold it level while the syrup goes in", _YAW_L),
        _wp(1.2, tcp0 - np.array([0.0, 0.0, CUP_SET_PRESS]), CUP_GRIP, "set it on the scale", _YAW_L),
        _wp(1.5, tcp0 - np.array([0.0, 0.0, CUP_SET_PRESS]), OPEN, "let go", _YAW_L),
        _wp(1.5, tcp0 - WITHDRAW * ap, OPEN, "withdraw", _YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": left_plan(), "right": right_plan()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
