"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights are above the table top unless named *_W.

A stack of three knocked-down cartons lies on a fixed brown board at x 270 mm, centred on y 0. An empty 12 x 8 x 6 in
carton (flaps already cut off) lies on the stack's top on one of its long sides, its open ends along x, so it stands
203 mm tall as a four-panel sleeve. Its corners are creases (hinges about axes along x). The fourth corner is the
manufacturer's glue joint: a 38 mm tab creased on the end panel's edge (the crease "tab") and glued to the
underneath panel (a FixedJoint, mechkit bond, closing the loop), so the sleeve is a closed four-bar with one degree
of freedom.

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

# -- the carton (loose, four panels, four creases, one glued) --------------------------------------------------
# A common 12 x 8 x 6 in single-wall C-flute shipping carton, flaps removed, lying on a long side (A).
# Board 3.2 mm (1/8 in, C-flute). Areal mass 0.55 kg/m2 (est, 200 lb test C-flute, about 0.5 to 0.6).
CT = 0.0032  # board thickness
LX = round(6 * IN, 4)  # 152.4 mm: the carton's length along x (its open ends face along x)
WA = round(12 * IN, 4)  # 304.8 mm: the long panels A (underneath) and C (on top)
WB = round(8 * IN, 4)  # 203.2 mm: the short panels B (+y end) and D (-y end), the carton's standing height
AREAL = 0.55
MASS_AC = round(LX * WA * AREAL, 4)  # 0.0255 kg each
MASS_BD = round(LX * WB * AREAL, 4)  # 0.0170 kg each
CARTON_MASS = round(2 * MASS_AC + 2 * MASS_BD, 3)  # 0.085 kg
CARTON_FRICTION = (0.50, 0.45)  # corrugated board on board (est)
CREASE_FRICTION = 0.010  # N m: a scored C-flute crease turns against about 0.01 N m (est)
CREASE_DAMPING = 0.02  # N m s/rad
CREASE_LIMIT = math.pi / 2 + 0.12  # +-96.9 degrees: an interior angle of -7 to 187 degrees
BOARD_BROWN = (0.62, 0.47, 0.30)
CARTON_BROWN = (0.72, 0.55, 0.34)
PILE_BROWN = (0.66, 0.50, 0.32)

# -- the stack of flats (loose) on its board (fixed) ------------------------------------------------------------------
BOARD_Y, BOARD_X, BOARD_T = 0.540, 0.175, 0.003  # a brown board, 540 (y) x 175 (x) x 3 mm
PILE_X, PILE_Y = 0.270, 0.0  # the stack's centre
FLAT_Y = round(WA + WB, 4)  # 508.0 mm: a knocked-down 12 x 8 x 6 in carton, two layers
FLAT_X = LX
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
TAB_W = round(1.5 * IN, 4)  # the manufacturer's glue tab: 1-1/2 in (38.1 mm) wide, 152.4 mm long, on A's top face
MASS_TAB = round(LX * TAB_W * AREAL, 4)
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
BODIES = ("base", "ab", "bc", "cd", "tab")
CREASES = ("ab", "bc", "cd", "tab")

# -- the rack: one angle PHI (0 standing, 90 degrees flat) --------------------------------------------------------- --
# Joint positions (rad, right-handed about +x): ab -PHI, bc +PHI, cd -PHI, tab +PHI when racked toward +y
# (they sum to 0: the glue tab stays flat on A, which is what lets the bond close the loop).
# Interior angle of a crease = 90 degrees minus its joint position (in degrees).


def joints_for(phi, sign=1.0):
    """Joint positions of the carton racked by ``phi`` toward +y (sign +1) or -y (sign -1)."""
    return {"ab": -sign * phi, "bc": sign * phi, "cd": -sign * phi, "tab": sign * phi}


def interior_deg(q):
    return 90.0 - math.degrees(q)


# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------- --
FLAT_HINGE_DEG = 15.0  # each crease within 15 degrees of fully shut (0) or fully open (180)
THICK_MAX = 0.012  # flat: highest point over lowest 12 mm or less (two layers are 6.4 mm)
FOOT_Y, FOOT_X = 0.540, 0.180  # flat: footprint within 540 x 180 mm (flat is 516 x 152)
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
STILL_MOVE = 0.001
STILL_TURN_DEG = 0.3
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
