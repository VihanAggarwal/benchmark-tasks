"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W, and
heights named h_* are above the LOW pad's top.

Two fixed 4x4 pads (89 x 89 mm, 300 mm long along x) carry a loose 2x4 sleeper (38 x 89 x 610 mm) standing on edge
along y. The floor dips under the near pad: its top is 6.0 mm below the far pad's. So the sleeper, resting on the
low pad's outer edge and the far pad's inner edge, slopes down toward +y. A wood shim (8 in long, 1-1/2 in wide,
3/8 in tapering to 1/16 in) lies on the low pad along x, its thin tip at the sleeper's front face. A 9 in torpedo
level lies along the sleeper's top.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
IN = 0.0254
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the pads (fixed) ----------------------------------------------------------------------------------------------
# 4x4 offcuts: a nominal 4x4 is 3-1/2 x 3-1/2 in (89 x 89 mm, American Softwood Lumber Standard PS 20 dressed size).
PAD_W = round(3.5 * IN, 4)  # 88.9 mm, across (along y)
PAD_LX = 0.300  # 300 mm long along x (offcut length, est)
PAD_X0, PAD_X1 = 0.100, 0.400  # the pads' front and back faces
PAD_X = (PAD_X0 + PAD_X1) / 2
FAR_H = PAD_W  # the far pad's top: 88.9 mm above the table
DIP = 0.0060  # the floor dips 6.0 mm under the low pad (spec): its top is 6.0 mm below the far pad's
LOW_H = FAR_H - DIP  # 82.9 mm: the low pad's block is 82.9 mm tall (it stands for a 4x4 on a floor 6 mm lower)
LOW_Y = -0.010  # the low pad's centre line
FAR_Y = -0.380  # the far pad's centre line
LOW_Y0, LOW_Y1 = LOW_Y - PAD_W / 2, LOW_Y + PAD_W / 2  # -0.05445 .. 0.03445
FAR_Y0, FAR_Y1 = FAR_Y - PAD_W / 2, FAR_Y + PAD_W / 2  # -0.42445 .. -0.33555
PAD_BROWN = (0.50, 0.40, 0.28)

# -- the sleeper (loose) -------------------------------------------------------------------------------------------
# A 2x4 is 1-1/2 x 3-1/2 in dressed (38 x 89 mm, PS 20). Cut to 610 mm (2 ft). Spruce-pine-fir, specific gravity
# 0.42 (AWC NDS Supplement), about 420 kg/m3 dry: 0.87 kg. Its 3 mm eased edges are left square in the model.
SL_W = round(1.5 * IN, 4)  # 38.1 mm across (along x)
SL_H = round(3.5 * IN, 4)  # 88.9 mm tall: on edge
SL_L = 0.610  # along y
SL_X = 0.300
SL_Y1 = 0.170  # its low (+y) end
SL_Y0 = SL_Y1 - SL_L  # -0.440: its far end
SL_Y = (SL_Y0 + SL_Y1) / 2  # -0.135
SL_DENSITY = 420.0
SL_MASS = round(SL_W * SL_H * SL_L * SL_DENSITY, 3)  # 0.868 kg
SL_FRONT_X = SL_X - SL_W / 2  # 0.28095: the face toward the robot (where the shim goes in)
SL_REAR_X = SL_X + SL_W / 2
SPF = (0.86, 0.75, 0.56)

# -- the shim (loose) ----------------------------------------------------------------------------------------------
# Wood shim, 8 in long, 1-1/2 in wide, 3/8 in at the butt tapered to 1/16 in (shim bundle listing, README).
SH_L = round(8 * IN, 4)  # 203.2 mm
SH_W = round(1.5 * IN, 4)  # 38.1 mm
SH_BUTT = round(0.375 * IN, 6)  # 9.525 mm
SH_TIP = round(IN / 16, 7)  # 1.5875 mm
SH_SLOPE = (SH_BUTT - SH_TIP) / SH_L  # 0.0391 mm per mm
SH_DENSITY = 420.0  # kiln-dried softwood (est)
SH_MASS = round(SH_W * SH_L * (SH_BUTT + SH_TIP) / 2 * SH_DENSITY, 3)  # 18 g
SH_Y = LOW_Y  # on the low pad's centre line, along x
SH_START_GAP = 0.002  # its tip starts 2 mm short of the sleeper's front face
SH_TIP_X0 = SL_FRONT_X - SH_START_GAP  # 0.27895
SH_BUTT_X0 = SH_TIP_X0 - SH_L  # 0.07575: the butt overhangs the pad's front face by 24 mm
SH_X0 = (SH_TIP_X0 + SH_BUTT_X0) / 2
SHIM_WOOD = (0.95, 0.88, 0.70)
SH_SLAB = 0.0014  # the sloped top as one turned slab 1.4 mm thick; flat blocks fill under it
SH_FILLERS = 8

# -- the level (loose) ---------------------------------------------------------------------------------------------
# 9 in magnetic torpedo level (Stanley 42-264 class): 229 mm long, 1 in thick, 0.2 lb (retailer listings, README).
LV_L = 0.229
LV_W = 0.025
LV_H = 0.045  # est (the listings give length and thickness only)
LV_MASS = 0.091  # 0.2 lb
LV_X, LV_Y = SL_X, SL_Y  # centred on the sleeper's top
LV_YELLOW = (0.95, 0.74, 0.08)
VIAL_GREEN = (0.72, 0.90, 0.55)
MARK_BLACK = (0.08, 0.08, 0.08)
BUBBLE_WHITE = (0.97, 0.98, 0.95)
VIAL_L, VIAL_W, VIAL_T = 0.036, 0.012, 0.001  # the top-read vial window on the level's top face
BUBBLE_L, BUBBLE_W, BUBBLE_T = 0.008, 0.008, 0.0015
MARK_W = 0.0015
MARK_C = 0.00725  # the two marks' centres: their inner edges 6.5 mm from the vial's centre
VIAL_R = 1.5  # m: the bubble moves VIAL_R x the level's pitch (est: a sensitive vial)
BUBBLE_MAX = 0.012  # the bubble's travel each way (the vial's ends)
BUBBLE_MASS = 0.002

# -- the rest pose as found ----------------------------------------------------------------------------------------
# A rigid sleeper on two flat tops of different heights rests on the low pad's outer (+y) top edge and the far pad's
# inner (+y) top edge: its underside is the straight line through those two edges.
START_SLOPE = -DIP / (LOW_Y1 - FAR_Y1)  # -0.01622: down toward +y
ROLL0 = math.atan(START_SLOPE)  # -0.929 deg about x (positive turns +y toward +z)
FRICTION = (0.45, 0.40)  # dry softwood on softwood (est), every wood part and the pads
LEVEL_FRICTION = (0.40, 0.35)  # ABS on softwood (est)


def h_line(y, slope, pivot_y, pivot_h=DIP):
    """Height (m) above the low pad's top of the sleeper's underside at ``y`` on the line through
    (pivot_y, pivot_h) with ``slope``."""
    return pivot_h + slope * (y - pivot_y)


def rest_on(a, y_contact):
    """(slope, pivot_y) of the sleeper resting on the far pad and on a support ``a`` m above the low pad's top at
    ``y_contact``: below level it bears on the far pad's inner edge, above level on its outer edge."""
    pivot = FAR_Y1 if a <= DIP else FAR_Y0
    return (a - DIP) / (y_contact - pivot), pivot


def underside_centre_z(slope, pivot_y):
    """Height above the table of the underside's centre line at the sleeper's centre."""
    return LOW_H + h_line(SL_Y, slope, pivot_y)


def box_centre(slope, pivot_y):
    """(y, z above the table) of the sleeper box's centre for an underside line with ``slope``."""
    phi = math.atan(slope)
    zb = underside_centre_z(slope, pivot_y)
    return SL_Y - SL_H / 2 * math.sin(phi), zb + SL_H / 2 * math.cos(phi)


START_PIVOT = FAR_Y1
SL_C0 = box_centre(START_SLOPE, START_PIVOT)  # (y, z) at spawn


def shim_t(d):
    """The shim's thickness (m) under the sleeper's front face with its tip ``d`` m past it (the thickest point of
    the shim under the sleeper's 38 mm footprint): the taper while the butt is short of the face, the butt's full
    thickness while it passes under, 0 once it is through or the tip has not reached the face."""
    if d <= 0.0:
        return 0.0
    if d <= SH_L:
        return SH_TIP + SH_SLOPE * d
    if d <= SH_L + SL_W:
        return SH_BUTT
    return 0.0


def end_diff(slope):
    """End-to-end height difference of the sleeper's top (m; + when the +y end is higher)."""
    return SL_L * math.sin(math.atan(slope))


def shim_bear(d):
    """The shim's thickness (m) the sleeper bears on, with its tip ``d`` m past the front face. The sleeper's flat
    underside lies flush on the sloped shim across its 38 mm width (GPU: it turns about y by the taper's 2.2 degrees),
    so its height is set by the shim's thickness at the sleeper's REAR face, not the front edge: ``shim_t`` less the
    taper over the 38 mm. Before the tip is past the rear face it bears on the front edge (``shim_t``); once the butt
    is under the footprint the butt's full thickness."""
    if d <= SL_W or d > SH_L:
        return shim_t(d)
    return SH_TIP + SH_SLOPE * (d - SL_W)


def diff_for_d(d):
    """The sleeper's end-to-end difference (m) resting on the shim pushed ``d`` past the front face: it bears on the
    shim's +y edge when low, its -y edge when high (or on the pad's outer edge when the shim does not reach)."""
    a = shim_bear(d)
    if a <= 0.0:
        return end_diff(START_SLOPE)
    y_c = SH_Y + SH_W / 2 if a < DIP else SH_Y - SH_W / 2
    slope, _ = rest_on(a, y_c)
    return end_diff(slope)


D_LEVEL = SL_W + (DIP - SH_TIP) / SH_SLOPE  # 151.0 mm: the tip travel past the front face that puts the sleeper level

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
LEVEL_TOL = 0.0010  # 1.0: end-to-end within 1 mm over 610 mm
NEAR_TOL = 0.0020  # 0.5: within 2 mm
LV_ON_GAP = 0.003  # the level's underside within 3 mm above the sleeper's top
LV_ON_ALIGN_DEG = 10.0  # its long axis within 10 degrees of the sleeper's
LV_ON_EDGE = 0.002  # its underside's centre over the sleeper's top face (2 mm allowed)
ON_PAD_LOW = 0.005  # the sleeper's lowest point no lower than 5 mm below the low pad's top (its overhanging end)
SHIM_ON_PAD = 0.005  # the shim's lowest point within 5 mm of the low pad's top
SHIM_FLAT_DEG = 15.0
FALL_DROP = 0.020  # latched: the sleeper or the level lower than 20 mm below the low pad's top (off the pads)
TIP_DEG = 30.0  # latched: the sleeper turned more than 30 degrees off its edge (onto its face)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001
STILL_TURN_DEG = 0.3
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
