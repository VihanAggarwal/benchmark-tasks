"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A Turck BI2-M12-AP6X-H1141 inductive sensor (M12 x 1 barrel, 52 mm long, 31 g, rated switching distance Sn 2 mm)
lies along x in an aluminium split clamp block (original part, 22 x 45 x 25 mm) closed by a KIPP K0006.1501406X20
adjustable cam lever (M6 x 20, lever A 70.4 mm, red RAL 3003), its face 12.0 mm short of a steel trip dog. The
setup sheet asks for a 1.0 mm face-to-dog gap (half of Sn; the maker's secured operating distance is 0.81 x Sn =
1.62 mm). An 8 in (203 mm) 1.00 mm feeler leaf, in a handle, lies on its edge in a rack tray beside the work; it is
slid in from the side.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)

# -- the dog (fixed): St37 trip dog on a carriage parked at its end stop ----------------------------------------------
X_DOG = 0.318  # the dog's face: the datum every gap is measured to
DOG_T = 0.004  # est: a 4 mm plate
DOG_Y, DOG_Z0, DOG_Z1 = 0.024, 0.0005, 0.0245  # 24 x 24 mm face (est), centred on the sensor axis
CAR_X1 = X_DOG + DOG_T + 0.040  # a 40 mm long carriage block behind it
CAR_W, CAR_H = 0.040, 0.034  # 40 mm wide, 34 mm tall (est); parked: fixed (the machine's slide spring is not modelled)
STEEL = (0.62, 0.64, 0.68)
DOG_STEEL = (0.45, 0.47, 0.52)
ALU = (0.78, 0.79, 0.80)

# -- the sensor (a slider link in the clamp) ------------------------------------------------------------------------
AXIS_Z = 0.0125  # the sensor's axis height above the table (the bore sits low in the block: est; half the leaf's width)
S_LEN = 0.052  # Turck datasheet "Dimensions 52 mm"
S_R = 0.006  # M12 x 1 barrel
CONN_L, CONN_R = 0.012, 0.007  # the M12 male connector at the rear (est)
S_MASS = 0.031  # Turck product page: 31 g
GAP0 = 0.012  # as found: the face 12.0 mm short of the dog's face
GAP_SET = 0.0010  # the setup sheet: half of Sn
SN = 0.002  # rated switching distance
SN_SECURED = 0.81 * SN  # 1.62 mm: the maker's secured operating distance
S_FACE0 = X_DOG - GAP0  # the face at q = 0
S_REAR0 = S_FACE0 - S_LEN
SLIDE_UP = GAP0  # the face can reach the dog (g = 0)
SLIDE_LO = -0.020
SENSOR_YELLOW = (0.80, 0.78, 0.30)
BRASS = (0.78, 0.66, 0.28)
CONN_BLACK = (0.12, 0.12, 0.13)
SLIDE_K = 2000.0  # N/m: the drive that follows the sensor each step when open (a drag) and holds it when shut
SLIDE_DAMP = 6.0  # N s/m

# -- the clamp (original part): a 6061 aluminium split block, 22 mm along the bore x 45 mm across x 25 mm tall, a
# 12.2 mm bore along x on the sensor's axis, split at the axis plane into a base and a 12.5 mm cap; the cap is held on
# its +y side by an M6 socket screw left just loose (a hinge) and closed on its -y side by a KIPP K0006.1501406X20
# adjustable cam lever (M6 x 20 stud into the base, A 70.4 mm, B 21.5 mm, aluminium red RAL 3003), its thrust piece set
# so the cam clamps the barrel as the lever comes down flat. Bolted to the machine bed by two M5 screws (not drawn).
CL_X0, CL_X1 = X_DOG - 0.038, X_DOG - 0.016  # 22 mm along the bore
CL_Y = 0.045  # 45 mm across
CL_H = 0.025  # 25 mm tall: 6.3 mm of metal under and over the 12.2 mm bore centred 12.5 mm up
BORE = 0.0124  # drawn as a 12.4 mm square bore round the 12 mm barrel (the real bore is 12.2 mm round)
BLOCK_ALU = (0.70, 0.71, 0.73)  # the block: bare aluminium
LEVER_PIVOT = (0.5 * (CL_X0 + CL_X1), -CL_Y / 2, CL_H)  # the cam's pin over the block's -y top edge, axis along x
LEVER_L = 0.0704  # KIPP K0006 size 1: A = 70.4 mm, the pin to the handle's end
LEVER_W, LEVER_T = 0.012, 0.006  # the handle drawn 12 mm wide, 6 mm thick (est)
LEVER_OPEN = math.radians(70.0)  # elevation as found
LEVER_MAX = LEVER_OPEN  # travel to horizontal (closed)
LEVER_CLOSED_BAND = math.radians(5.0)
LEVER_REARM = math.radians(20.0)  # the lever counts as re-opened once 20 degrees or more off its shut stop
LEVER_MASS = 0.020  # est: an aluminium size 1 cam lever
LEVER_RED = (0.72, 0.12, 0.12)  # RAL 3003 ruby red
LEVER_SPRING = 0.06  # N m/rad: the detent's pull toward the notch it is near
LEVER_CAPTURE = 0.35  # rad
LEVER_DAMP = 0.02
LEVER_FRICTION = 0.003

# -- the feeler leaf (loose): 1.00 mm leaf, 12.7 mm wide (1/2 in feeler stock), 8 in long, in a handle -----------
LF_T = 0.0010  # 1.00 mm (est: the setup sheet's gauge)
LF_W = 0.0127  # 1/2 in, as Starrett 667 series feeler stock
LF_L = 0.203  # 8 in leaf (est length; the 667 series stock is cut from 12 in)
LF_H_X, LF_H_Y, LF_H_Z = 0.022, 0.040, 0.024  # handle: 22 across the grip (along x), 40 long, 24 tall (est)
LF_H_YC = -0.198  # the handle's centre along the leaf (the tip is at 0): 25 mm of the leaf sits in it
LF_MASS = 0.030  # est: 20 g steel leaf + a 10 g ABS handle
LF_STEEL = (0.74, 0.76, 0.80)
LF_ORANGE = (0.90, 0.45, 0.10)
LF_FRICTION = (0.6, 0.5)  # steel on steel, dry (est)

# -- the rack tray (fixed): the leaf lies in it along y, on its handle ------------------------------------------
TRAY_X, TRAY_Y = 0.230, -0.245
TRAY_OUT = (0.060, 0.270, 0.008)  # outer x, y, z (est)
TRAY_WALL, TRAY_FLOOR = 0.003, 0.002
TRAY_GREEN = (0.30, 0.50, 0.35)
LEAF_X0, LEAF_Y0 = TRAY_X, -0.115  # the leaf's tip as found (the handle then at y -0.313)
LEAF_Z0 = TRAY_FLOOR + 0.0002
GAP_AIM_X = X_DOG - 0.0035  # where the demo aims the leaf's centre in the gap: 3 mm off the dog, 8 mm off the face
GAP_AIM_TIP_Y = 0.012  # the leaf's tip this far past the sensor's axis: over the face and the dog's width

# -- thresholds (scoring.py) ----------------------------------------------------------------------------------------
GOOD_TOL = 0.00025  # 1.0 and 0.75: the gap within 1.0 +- 0.25 mm
RANGE_LO, RANGE_HI = 0.0005, SN_SECURED  # 0.5: inside the maker's secured range 0.5 .. 1.62 mm
POOR_LO, POOR_HI = 0.0003, 0.0040  # 0.25: 0.30 .. 4.0 mm
FILL_TOL = 0.00025  # locked on the leaf: the gap within 0.25 mm of the leaf's 1.00 mm as the lever comes shut
UPRIGHT_DEG = 10.0  # locked on the leaf: the blade square to the axis and its width vertical within 10 degrees
FACE_COVER = 0.006  # locked on the leaf: the blade over 6 mm or more of the face's 12 mm round (across, in y)
CRASH_GAP = 0.0003  # latched zero: the clamp shut with the face closer than 0.30 mm to the dog
OUT_OF_GAP = 0.030  # the leaf is out of the gap when every leaf point is 30 mm or more from the gap's centre line (xy)
IN_TRAY_LOW = 0.010  # lowest point within 10 mm of the tray's floor
LET_GO = 0.030
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001
STILL_TURN_DEG = 0.3
STILL_SLIDE = 0.0002  # the sensor's slide moves under 0.2 mm in the window
STILL_LEVER_DEG = 1.0
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
EPS = 1e-9

GAP_X = 0.5 * (S_FACE0 + X_DOG)  # the gap's centre line along x as found
GAP_Y = 0.0


def gap_of(q: float) -> float:
    """Face-to-dog gap (m) with the sensor slid ``q`` m toward the dog."""
    return X_DOG - (S_FACE0 + q)


def lever_face_point(q: float, s: float):
    """(y, z above the table) of the point ``s`` m from the pivot along the lever's inboard-facing top face (its face
    toward the block) at joint angle ``q`` (0 as found, LEVER_MAX shut)."""
    e = LEVER_OPEN - q
    py, pz = LEVER_PIVOT[1], LEVER_PIVOT[2]
    dy, dz = -math.cos(e), math.sin(e)
    ny, nz = math.sin(e), math.cos(e)
    return (py + s * dy + LEVER_T / 2 * ny, pz + s * dz + LEVER_T / 2 * nz)


def lever_tip_top(q: float):
    """(y, z above the table) of the lever's tip, top face centre, at joint angle ``q``."""
    return lever_face_point(q, LEVER_L)
