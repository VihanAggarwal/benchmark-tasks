"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

The workbench's back upstand (backboard) runs along the bench: a 30 mm board (front face at x 400 mm, 700 mm wide,
210 mm tall) held upright by two braces screwed to the bench top behind it, with the bench's end post (a 30 x 30 mm
upright, 210 mm tall) against it on the robot's left, 110 mm beyond the shelf's end. A French cleat rail is
screwed to the backboard: birch ply, 11 mm thick, 340 mm long, its top edge cut at 45 degrees so that it is 11 mm
lower at the board than at its front face. The loose shelf is a small tool ledge (open top and front, birch ply)
with the matching cleat screwed to its back. Hung, the unit cleat's bevel lies on the rail's bevel, its back face
on the backboard, the shelf level, 40 mm over the bench.

The unit frame (shelf): origin at the middle of the shelf's underside; x toward the wall, y left, z up.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
S2 = math.sqrt(2.0)

# -- the backboard ("wall", fixed) ------------------------------------------------------------------------------
WALL_X = 0.400  # the backboard's front face
WALL_T = 0.030  # 30 mm board (two 15 mm ply sheets, est)
WALL_W = 0.700  # y +-350 mm
WALL_H = 0.210  # a 210 mm upstand; the arms' wrists (link 4) pass over its top edge at the seat
BATTEN_X1 = 0.560  # two braces behind the board, screwed to the bench top, hold it upright
BATTEN_T, BATTEN_W = 0.030, 0.040
BATTEN_Y = 0.250
SIDE_Y0, SIDE_Y1 = 0.270, 0.300  # the bench's end post on the left: its inner face 270 mm from the midline
SIDE_X0 = 0.370  # it runs from the wall out to here (30 mm deep: clear of the arms' ready pose by 30 mm)
SIDE_H = 0.210
WALL_COLOR = (0.88, 0.87, 0.82)
SIDE_COLOR = (0.80, 0.78, 0.72)
BATTEN_COLOR = (0.60, 0.50, 0.36)

# -- the cleats (45 degrees) ---------------------------------------------------------------------------------------
CLEAT_T = 0.012  # unit cleat thickness (x): the shelf stands 12 mm off the wall
WCLEAT_T = 0.011  # wall cleat thickness: 1 mm shy of the unit cleat's front face so the shelf's back never touches it
WCLEAT_L = 0.340  # y +-170 mm
UCLEAT_L = 0.300  # y +-150 mm (on the shelf's back)
SEAT = 0.040  # hung: the shelf's underside 40 mm over the bench
WCLEAT_Z0 = 0.020  # wall cleat's bottom edge
BEV_BACK = 0.050  # the bevel's height at the wall face (world): 10 mm over the shelf's underside
RISE = CLEAT_T  # 45 degrees: 12 mm lower at the wall than at the unit cleat's front face
SLAB_T = 0.0025  # the bevel as a turned 2.5 mm slab; stairs of 1 mm fill under (wall) or over (unit)
STEP = 0.001
PLY = (0.86, 0.74, 0.52)  # birch ply
CLEAT_PLY = (0.80, 0.66, 0.44)
SCREW_GREY = (0.55, 0.55, 0.58)
FRICTION = (0.45, 0.40)  # birch ply on birch ply (est)
GRIP_LOCAL_X = -0.010  # where the fingers pinch the end panels (local x): 32 mm in front of the centre of mass, so the
# gripper's housing stays 28 mm short of the wall (the housing reaches 40 mm past the TCP)
SCREW_Y = (-0.120, 0.0, 0.120)  # three screws through the wall cleat into the wall
SCREW_R, SCREW_H, SCREW_Z = 0.004, 0.0015, 0.030  # 8 mm pan heads below the shelf's underside, 1.5 mm proud
USCREW_Y = (-0.100, 0.0, 0.100)  # three screws through the shelf's back panel into the unit cleat (heads inside)
USCREW_Z = 0.050  # their heads' height above the shelf's underside, on the back panel's inner face


def bevel_z(x):
    """World height (above the bench) of the bevel line at world x on the wall (also the unit cleat's lower edge when
    hung): 50 mm at the wall face, 12 mm higher at the unit cleat's front face (x 388 mm)."""
    return BEV_BACK + (WALL_X - x)


# -- the shelf (loose) ---------------------------------------------------------------------------------------------
# A small display ledge, open at the top and front: two 18 mm end panels, a 6 mm bottom and a 6 mm back (birch ply,
# about 680 kg/m3), a 12 mm cleat on its back. End panels thick enough to pinch (16 mm or more).
SH_D = 0.100  # depth (x)
SH_W = 0.320  # width (y)
SH_H = 0.080  # end panel height (z)
END_T = 0.018
BOT_T = 0.006
BACK_T = 0.006
INNER = SH_W / 2 - END_T  # 142 mm: the inside half width
UC_TOP = 0.070  # the unit cleat's top edge, above the shelf's underside
UC_FRONT_LOCAL = SH_D / 2  # the unit cleat's front face (local x 50 mm)
UC_BACK_LOCAL = SH_D / 2 + CLEAT_T  # 62 mm: its back face, on the wall when hung
PLY_DENSITY = 680.0
_UC_AREA = CLEAT_T * (UC_TOP - (BEV_BACK - SEAT + RISE / 2))  # mean cleat height times its thickness
SH_MASS = round(PLY_DENSITY * (2 * END_T * SH_D * SH_H + 2 * INNER * SH_D * BOT_T
                               + 2 * INNER * BACK_T * (SH_H - BOT_T) + _UC_AREA * UCLEAT_L), 3)
SH_START_X = 0.220  # the shelf's centre at the start (rear face of the box at x 270 mm)
SH_SEAT_X = WALL_X - UC_BACK_LOCAL  # 338 mm: the unit frame's x when hung
SH_FRONT_LOCAL = -SH_D / 2
END_MID = SH_W / 2 - END_T / 2  # 151 mm: the end panels' middle lines (where the fingers pinch)

# the bevel in the unit frame: local z of the unit cleat's lower edge at local x
def u_bevel_z(lx):
    return (BEV_BACK - SEAT) + (UC_BACK_LOCAL - lx)


# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
CONTACT_TOL = 0.0010  # bevel faces coincide within 1 mm (a 1 mm feeler will not go in)
CONTACT_X_A, CONTACT_X_B = UC_FRONT_LOCAL, UC_FRONT_LOCAL + 0.010  # the unit bevel's two probe points (local x)
SLOPE_OVERLAP = 0.004  # the bevel faces overlap by 4 mm or more along the slope
BACK_LO, BACK_HI = -0.0015, 0.0010  # seated: unit cleat's back face from 1.5 mm off the wall plane to 1 mm into it
BACK_HUNG = 0.008  # hung: back face no more than 8 mm off the wall (3 mm into it, a reading error)
UNDER_TOL = 0.002  # seated: the underside within 2 mm of 40 mm
LEVEL_DEG = 1.0  # seated: tilt about the long axis, back face off vertical
HUNG_DEG = 3.0
FULL_ENGAGE = UCLEAT_L - 0.0005  # the whole unit cleat over the wall cleat (within 0.5 mm)
HALF_ENGAGE = 0.150  # hung: at least half of it over the wall cleat
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001
STILL_TURN_DEG = 0.3
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9

# -- the wall's bevel in 2-D (x, z above the bench) --------------------------------------------------------------------
# The wall slab lies on the line below its top face: front corner at x 389 + SLAB_T/sqrt(2)... its lower front corner
# stays on the cleat's front face (x 389 mm), so its bevel face starts 1.8 mm behind it.
_D = SLAB_T / S2
W_FACE_X0 = WALL_X - WCLEAT_T + _D  # 390.8 mm: where the wall's bevel face starts
W_FACE_X1 = WALL_X  # and ends at the wall
U_FACE_X0 = WALL_X - CLEAT_T  # 388 mm: the unit's bevel face starts at its front face
U_FACE_X1 = WALL_X - _D  # 398.2 mm: and ends before its upper corner would enter the wall
PATCH = U_FACE_X1 - W_FACE_X0  # 7.4 mm of x where the two bevel faces meet


def plane_distance(x, z):
    """Signed distance (m) from a point (x, z above the bench) to the wall's bevel plane; + above it."""
    return ((x - WALL_X) + (z - BEV_BACK)) / S2


def slope_s(x, z):
    """Position along the bevel (m) of the point's projection: down the slope toward the wall is +."""
    return (x - z) / S2

PITCH_RAD = math.radians(45.0)  # the bevel's turn about y
