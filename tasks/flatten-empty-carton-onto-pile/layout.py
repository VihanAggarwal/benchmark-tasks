"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights are above the table top unless named *_W.

A stack of three knocked-down cartons lies on a fixed brown board at x 270 mm, centred on y 0. An empty 12 x 8 x 6 in
regular slotted carton (RSC) lies on the stack's top on one of its long sides, its open ends along x, so it stands
203 mm tall as a four-panel sleeve with four flaps at each open end. It was emptied through its near (-x) end, whose
four flaps were folded back and lie out straight; the tape on its far (+x) end has just been slit, and those four
flaps hang loose at uneven angles (FLAP_SET). Its corners are creases (hinges about axes along x). The fourth corner
is the manufacturer's glue joint: a 38 mm tab creased on the end panel's edge (the crease "tab") and glued to the
underneath panel (a FixedJoint, mechkit bond, closing the loop), so the sleeve is a closed four-bar with one degree of
freedom. Each flap is a rigid board plate on its own crease along its panel's open-end edge, drawn at its start angle
(its first set: where its crease spring rests); bent more than a few degrees, the crease takes a new set there.

The sleeve collapses flat by racking it toward +y: one parameter PHI turns all four creases by the same angle:
interior angles 90 + PHI at the near-bottom crease (A-B), 90 - PHI at the top (B-C), 90 + PHI at the crease on the
end panel (C-D), 90 - PHI at the glue tab; at PHI = 90 degrees the carton is two 3.2 mm layers on the stack,
6.4 mm thick, 516 mm long.
"""

import math

MM = 0.001
IN = 0.0254
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the carton (loose, four panels, four creases, one glued, eight flaps) -------------------------------------
# A common 12 x 8 x 6 in single-wall C-flute regular slotted carton (RSC), lying on a long side (A), flaps attached.
# Board 3.2 mm (1/8 in, C-flute). Areal mass 0.55 kg/m2 (est, 200 lb test C-flute, about 0.5 to 0.6).
CT = 0.0032  # board thickness
LX = round(6 * IN, 4)  # 152.4 mm: the carton's length along x (its open ends face along x)
WA = round(12 * IN, 4)  # 304.8 mm: the long panels A (underneath) and C (on top)
WB = round(8 * IN, 4)  # 203.2 mm: the short panels B (+y end) and D (-y end), the carton's standing height
AREAL = 0.55
MASS_AC = round(LX * WA * AREAL, 4)  # 0.0255 kg each
MASS_BD = round(LX * WB * AREAL, 4)  # 0.0170 kg each
# Flaps: an RSC's flaps are all half the carton's width deep (8 in / 2 = 4 in), so the long (major) flaps meet in the
# middle when closed (the RSC definition, FEFCO code 0201). Major flaps on the long panels A and C (305 x 102 mm),
# minor flaps on the end panels B and D (203 x 102 mm), one at each open end (-x "near", +x "far"). The slots between
# flaps (about 1/4 in) are left out: each flap is drawn as long as its panel.
FD = round(4 * IN, 4)  # 101.6 mm flap depth
MASS_MAJOR = round(WA * FD * AREAL, 4)  # 0.0170 kg each
MASS_MINOR = round(WB * FD * AREAL, 4)  # 0.0114 kg each
# A flap crease (est): springy for a few degrees about where it rests (its set: the start angle, as drawn, at first)
# and, bent further, it takes a new set where the flap is (scored board yields at the crease), so a flap stays where
# it is put. In the sim: a joint spring toward its set (a drive) and a per-step rule that moves the set (the drive's
# target) to the flap's angle once the flap is more than FLAP_YIELD_DEG from it (task.py). No joint friction: on the
# GPU (18 Nov) 0.03 N m of PhysX joint friction let the flaps creep 2 to 3 degrees a second with no drag, as the
# program's crank and wing-nut runs found for joint friction; a drive holds by its spring at the position level.
FLAP_SPRING = 0.30  # N m/rad toward the set
FLAP_YIELD_DEG = 3.0  # bent more than this from its set, the crease takes a new set there
FLAP_FRICTION = 0.0  # N m: no joint friction (see above)
FLAP_DAMPING = 0.02  # N m s/rad (critical is 2 (K I)^0.5, about 0.008 for a long flap: overdamped, no ring)
# The flap models a GPU variant job can compare (task.py, DEMO_VARIANT=<name>; "crease_set" is the submitted one):
#   crease_set      the above
#   crease_stiff    a stiffer, tighter crease: 1.0 N m/rad, sets past 1.5 degrees (holds 0.026 N m, not 0.016)
#   crease_ratchet  as crease_set, but the set only follows a flap opened toward straight, never one bent away
#   friction_hold   the 18 Nov model (spring 0.01 N m/rad toward the start angle, 0.03 N m friction, no set rule;
#                   GPU demo 0.25 from the creep); offline only, for comparison: not to be queued
FLAP_MODELS = {
    "crease_set": {"spring": FLAP_SPRING, "yield_deg": FLAP_YIELD_DEG, "damping": FLAP_DAMPING,
                   "friction": FLAP_FRICTION, "ratchet": False},
    "crease_stiff": {"spring": 1.0, "yield_deg": 1.5, "damping": 0.03, "friction": 0.0, "ratchet": False},
    "crease_ratchet": {"spring": FLAP_SPRING, "yield_deg": FLAP_YIELD_DEG, "damping": FLAP_DAMPING,
                       "friction": 0.0, "ratchet": True},
    "friction_hold": {"spring": 0.01, "yield_deg": None, "damping": 0.02, "friction": 0.03, "ratchet": False},
}
FLAP_MODEL_DEFAULT = "crease_set"
FLAP_LIMIT = math.pi / 2 + 0.12  # +-96.9 degrees from straight: folded shut (90) and a little past, either way
TAB_W = round(1.5 * IN, 4)  # the manufacturer's glue tab: 1-1/2 in (38.1 mm) wide, 152.4 mm long, on A's top face
MASS_TAB = round(LX * TAB_W * AREAL, 4)
CARTON_MASS = round(2 * MASS_AC + 2 * MASS_BD + 4 * MASS_MAJOR + 4 * MASS_MINOR + MASS_TAB, 3)  # 0.202 kg in all
CARTON_FRICTION = (0.50, 0.45)  # corrugated board on board (est)
CREASE_FRICTION = 0.010  # N m: a scored C-flute crease turns against about 0.01 N m (est)
CREASE_DAMPING = 0.02  # N m s/rad
CREASE_LIMIT = math.pi / 2 + 0.12  # +-96.9 degrees: an interior angle of -7 to 187 degrees
BOARD_BROWN = (0.62, 0.47, 0.30)
CARTON_BROWN = (0.72, 0.55, 0.34)
PILE_BROWN = (0.66, 0.50, 0.32)

# -- the stack of flats (loose) on its board (fixed) ------------------------------------------------------------------
BOARD_Y, BOARD_X, BOARD_T = 0.540, 0.380, 0.003  # a brown board, 540 (y) x 380 (x) x 3 mm
PILE_X, PILE_Y = 0.270, 0.0  # the stack's centre
FLAT_Y = round(WA + WB, 4)  # 508.0 mm: a knocked-down 12 x 8 x 6 in carton, two layers
FLAT_X = round(LX + 2 * FD, 4)  # 355.6 mm: its 6 in depth plus a 4 in flap out straight at each end
FLAT_T = 2 * CT  # 6.4 mm
FLAT_N = 3
FLAT_MASS = CARTON_MASS
FLAT_FRICTION = (0.50, 0.45)
PT = BOARD_T + FLAT_N * FLAT_T  # 22.2 mm: the stack's top above the table


def flat_z(i):
    """Height above the table of the centre of flat ``i`` (0 is the bottom one)."""
    return BOARD_T + (i + 0.5) * FLAT_T


# -- where the carton starts: lying on the stack's top, its glue tab at the -y end -----------------------------
CARTON_Y = -0.1016  # the long panel A's centre in y: A spans y -254.0 to +50.8 mm (from the stack's -y end)
CARTON_X = PILE_X
A_YMIN = CARTON_Y - WA / 2  # -254.0 mm
A_YMAX = CARTON_Y + WA / 2  # +50.8 mm
CARTON_ORIGIN_Z = PT + CT / 2  # A's centre height above the table

# Panel shapes in the carton's base frame (origin at A's centre, creases drawn standing at 90 degrees).
# Each crease is drawn offset one board thickness so the panels lie in two clean layers when flat.
ZB = WB + CT  # panel B stands from z 0 to 206.4 mm (its top reaches the top panel's mid-plane)
Z_BC = WB  # crease B-C height (on B's top)
Z_C = WB + CT  # the top panel's mid-plane height
LD = WB + 1.5 * CT  # panel D hangs from C's -y end to half a board below A's mid-plane (the stack's top)
CREASE_AB = (0.0, WA / 2, 0.0)
CREASE_BC = (0.0, WA / 2, Z_BC)
CREASE_CD = (0.0, -WA / 2, Z_C)
CREASE_TAB = (0.0, -WA / 2, CT)  # D's foot: WB below crease C-D, as B is WB long: a four-bar
PANELS = {  # body -> (centre, size, mass), all in the base frame as drawn
    "base": ((0.0, 0.0, 0.0), (LX, WA, CT), MASS_AC),
    "ab": ((0.0, WA / 2, ZB / 2), (LX, CT, ZB), MASS_BD),
    "bc": ((0.0, 0.0, Z_C), (LX, WA, CT), MASS_AC),
    "cd": ((0.0, -WA / 2, Z_C - LD / 2), (LX, CT, LD), MASS_BD),
    "tab": ((0.0, -WA / 2 + TAB_W / 2, CT), (LX, TAB_W, CT), MASS_TAB),
}
ANCHOR = {"base": (0.0, 0.0, 0.0), "ab": CREASE_AB, "bc": CREASE_BC, "cd": CREASE_CD, "tab": CREASE_TAB}
PARENT = {"ab": "base", "bc": "ab", "cd": "bc", "tab": "cd"}
CREASES = ("ab", "bc", "cd", "tab")  # the four corner creases (the tab's is the glue joint's)

# The eight flaps, straight out here (task.py draws, and scoring.py reads, each turned to its set, FLAP_SET below):
# each on a crease along its panel's open-end edge (x = -+76.2 mm), the
# crease's axis along y for the long panels' flaps (A, C: they lie level) and along z for the end panels' (B, D: they
# stand upright). Name: panel, then n (near, -x) or f (far, +x).
FLAP_OF = {"base": "a", "ab": "b", "bc": "c", "cd": "d"}
FLAP_AXIS = {"base": "Y", "ab": "Z", "bc": "Y", "cd": "Z"}
FLAPS = tuple(f"{FLAP_OF[p]}{e}" for p in ("base", "ab", "bc", "cd") for e in ("n", "f"))
for _panel in ("base", "ab", "bc", "cd"):
    (_cx, _cy, _cz), (_sx, _sy, _sz), _ = PANELS[_panel]
    for _end, _s in (("n", -1.0), ("f", 1.0)):
        _name = f"{FLAP_OF[_panel]}{_end}"
        PANELS[_name] = ((_s * (LX / 2 + FD / 2), _cy, _cz), (FD, _sy, _sz),
                         MASS_MAJOR if _panel in ("base", "bc") else MASS_MINOR)
        ANCHOR[_name] = (_s * LX / 2, _cy, _cz)
        PARENT[_name] = _panel
AXIS = {**{k: "X" for k in CREASES}, **{f"{FLAP_OF[p]}{e}": FLAP_AXIS[p] for p in FLAP_AXIS for e in ("n", "f")}}
BODIES = ("base", "ab", "bc", "cd", "tab", *FLAPS)
JOINTS = (*CREASES, *FLAPS)

# -- where each flap starts (its set): the angle from straight out, signed about its crease's axis (+Y or +Z) ------- --
# The near (-x) end: the carton was emptied through it, its flaps folded back hard, and they lie out straight (0).
# The far (+x) end: its tape has just been slit; the four flaps hang loose at uneven angles (real trial: in ranges):
#   cf  the top panel's long flap droops 45 degrees below level, over the opening (real 40 to 55)
#   af  the underneath panel's long flap stands 30 degrees up off the stack (real 20 to 40)
#   df  the -y end panel's short flap hangs 30 degrees out past straight, away from the carton (real 20 to 40)
#   bf  the +y end panel's short flap is within 10 degrees of straight (sim: straight)
# Signs: about +Y a positive angle turns a far flap's tip down (x toward -z); about +Z, toward +y.
FLAP_START_DEG = {"cf": 45.0, "af": 30.0, "df": 30.0}
FLAP_SET = {k: 0.0 for k in FLAPS}
FLAP_SET.update(cf=math.radians(45.0), af=-math.radians(30.0), df=-math.radians(30.0))
LOOSE = ("cf", "af", "df")  # the far end's flaps that do not start straight
X_FAR = LX / 2  # every far flap's crease, in the base frame (x 346.2 mm in the world)


def flap_joint(k, off):
    """The joint position of flap ``k`` (0 as drawn: its set) at ``off`` rad from straight, signed as FLAP_SET."""
    return off - FLAP_SET[k]


def flap_off(k, q):
    """Flap ``k``'s angle from straight out (rad, signed as FLAP_SET) at joint position ``q``."""
    return q + FLAP_SET[k]

# -- the rack: one angle PHI (0 standing, 90 degrees flat) --------------------------------------------------------- --
# Joint positions (rad, right-handed about +x): ab -PHI, bc +PHI, cd -PHI, tab +PHI when racked toward +y
# (they sum to 0: the glue tab stays flat on A, which is what lets the bond close the loop).
# Interior angle of a crease = 90 degrees minus its joint position (in degrees).


def joints_for(phi, sign=1.0, flaps="straight"):
    """Joint positions of the carton racked by ``phi`` toward +y (sign +1) or -y (sign -1), every flap straight out
    (``flaps`` "straight"), at its start angle ("start"), or at the angles from straight in a dict (flap -> rad; the
    rest straight)."""
    q = {"ab": -sign * phi, "bc": sign * phi, "cd": -sign * phi, "tab": sign * phi}
    off = {k: 0.0 for k in FLAPS} if flaps == "straight" else dict(FLAP_SET) if flaps == "start" else \
        {k: float(flaps.get(k, 0.0)) for k in FLAPS}
    q.update({k: flap_joint(k, off[k]) for k in FLAPS})
    return q


def interior_deg(q):
    return 90.0 - math.degrees(q)


# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------- --
FLAT_HINGE_DEG = 15.0  # each crease within 15 degrees of fully shut (0) or fully open (180); each flap of straight
THICK_MAX = 0.012  # flat: highest point over lowest 12 mm or less (two layers are 6.4 mm)
FOOT_Y, FOOT_X = 0.540, 0.380  # flat: footprint within 540 x 380 mm (flat, flaps out, is 516 x 356)
THICK_HALF = 0.110  # 0.25: collapsed more than half: no more than 110 mm tall
ON_CENTRE = 0.060  # on the stack: footprint's centre within 60 mm of the stack's centre (plan)
ON_LOW = 0.006  # its lowest point within 6 mm of the stack's top
ON2_CENTRE, ON2_LOW = 0.120, 0.020  # 0.75: sitting on it, offset or high on one side
KEEP_MOVE = 0.015  # stack kept: no flat moves over 15 mm (plan) or 4 mm up, or turns over 8 degrees
KEEP_LIFT = 0.004
KEEP_TURN_DEG = 8.0
KNOCK_MOVE = 0.040  # latched: any flat moved over 40 mm or turned over 30 degrees
KNOCK_TURN_DEG = 30.0
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # still: over the last 1 s no panel, flap or flat moves 5 mm or turns 5 degrees (no visible motion)
STILL_TURN_DEG = 5.0
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
