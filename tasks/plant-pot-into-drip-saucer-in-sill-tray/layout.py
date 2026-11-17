"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top (z 0.7475) is the window sill's top surface.

Scene, back to front: a window (glass in a white frame) at the sill's back; a black sill tray with two bays, in the
right-hand bay a loose drip saucer, in the left-hand bay another household plant (its saucer, pot, stem and a wide
canopy of leaves) that must be left alone; and, near the right arm, a sink drainer tray with the watered, drained
pot standing in it. The pot (3 in square nursery pot, stepped sides standing in for the taper) is carried from the
drainer and seated in the saucer.
"""

import math

TOP = 0.7475  # station table top = the sill's top surface

# -- the sill tray (fixed): a shallow black plastic tray with a raised edge and two bays formed by low stops ----------
TRAY_X, TRAY_Y = 0.33, 0.0  # centre of the tray
BAY = 0.102  # a bay's inside square (a saucer's 92 mm base sits in it with 5 mm each side)
BAY_PITCH = 0.132  # between the two bays' centres (along y)
BAY_B = (TRAY_X, -BAY_PITCH / 2)  # the right-hand bay (the saucer's)
BAY_A = (TRAY_X, +BAY_PITCH / 2)  # the left-hand bay (the neighbouring plant's)
STOP_T, STOP_H = 0.003, 0.004  # the stops: 3 mm wide, 4 mm tall on the floor, all round each bay
TRAY_FLOOR = 0.003  # the tray's floor thickness: its top is 3 mm above the table
TRAY_WALL = 0.003
TRAY_H = 0.012  # the raised edge: 12 mm above the table
TRAY_INNER = (2 * (BAY / 2 + STOP_T + 0.010), BAY_PITCH + 2 * (BAY / 2 + STOP_T + 0.010))  # x, y inside the edge
TRAY_OUTER = (TRAY_INNER[0] + 2 * TRAY_WALL, TRAY_INNER[1] + 2 * TRAY_WALL)  # 0.134 x 0.266
TRAY_BLACK = (0.10, 0.10, 0.11)

# -- the drip saucer (loose): square, 98 mm at the rim, stepped walls ------------------------------------------------
SAU_BASE = 0.092  # outside width of the floor and the lower wall (what stands in the bay)
SAU_RIM = 0.098  # outside width of the flared upper wall
SAU_H = 0.012  # height
SAU_FLOOR = 0.003  # floor thickness: the well's floor is 3 mm above the saucer's underside
SAU_STEP = 0.008  # height of the lower wall's top
SAU_WELL = 0.072  # inside width of the well at its floor (lower wall)
SAU_MOUTH = 0.084  # inside width of the upper wall
SAU_MASS = 0.045
SAU_TERRACOTTA = (0.74, 0.36, 0.22)

# -- the pot (loose): a 3 in square nursery pot with damp soil and a small succulent --------------------------------
POT_BASE = 0.058  # outside width at the foot
POT_LAYERS = ((0.000, 0.024, 0.058), (0.024, 0.048, 0.065), (0.048, 0.072, 0.072))  # z0, z1, width: the taper
POT_RIM_Z0, POT_H, POT_RIM = 0.072, 0.078, 0.078  # a 6 mm rim ring, 78 mm outside, 3 mm walls
POT_SOIL_TOP = 0.074
PLANT_R, PLANT_H = 0.022, 0.012  # the succulent rosette on the soil: 12 mm high, 44 mm across
POT_TOP = POT_SOIL_TOP + PLANT_H  # 0.086
POT_MASS = 0.20
POT_BLACK = (0.08, 0.08, 0.09)
SOIL_BROWN = (0.25, 0.16, 0.10)
PLANT_GREEN = (0.30, 0.55, 0.30)
GRIP_Z0, GRIP_Z1 = 0.024, 0.048  # the band the fingers hold: the second layer, 65 mm wide
GRIP_W = 0.065

# -- the neighbouring plant (loose, one body): saucer, pot, stem and a canopy of leaves ------------------------------
NB_STEM = 0.008  # the stem's square section
NB_CANOPY_R, NB_CANOPY_T = 0.080, 0.006
NB_CANOPY_Z = 0.150  # centre of the canopy above the neighbour saucer's underside
NB_MASS = 0.30
LEAF_GREEN = (0.20, 0.45, 0.22)

# -- the sink drainer (fixed): a shallow tray beside the sink; the watered pot stands in it ------------------------
DRAIN_OUTER = 0.130
DRAIN_WALL = 0.003
DRAIN_FLOOR = 0.003
DRAIN_H = 0.015  # the edge: 15 mm above the table
POT_X0, POT_Y0 = 0.20, -0.235  # the pot's start (centre of its foot) = the drainer's centre
DRAIN_GREY = (0.62, 0.64, 0.66)

# -- the window (fixed): glass in a white frame behind the tray -------------------------------------------------------
WIN_X = 0.455  # centre of the glass
GLASS_T = 0.006
GLASS_W, GLASS_Z0, GLASS_Z1 = 0.66, 0.040, 0.340
FRAME_DEPTH, FRAME_T = 0.040, 0.040  # rails and jambs: 40 mm deep (x), 40 mm wide
FRAME_WHITE = (0.93, 0.93, 0.90)
GLASS_BLUE = (0.72, 0.84, 0.92)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SEAT_LOW, SEAT_HIGH = -0.002, 0.003  # the pot's foot corners over the saucer's well floor: within 2 mm into, 3 mm above
SEAT_TILT_DEG = 5.0  # the pot's axis against the saucer's
SEAT_CENTRE = 0.008  # the pot's axis within 8 mm of the saucer's axis (top view)
SEAT_YAW_DEG = 10.0  # the pot's sides within 10 degrees of the saucer's (a square into a square)
WELL_SLACK = 0.001  # the foot's corners inside the well's 72 mm square, 1 mm allowed
SAU_FLAT_DEG = 3.0  # 'the saucer lies flat'
SAU_FLOOR_TOL = 0.002  # its underside within 2 mm of the tray floor's top (not on a stop)
BAY_SLACK = 0.001  # the base's corners inside the bay's square, 1 mm allowed (a ruler can read it)
POT_UPRIGHT_DEG = 10.0  # 'standing' for the 0.25 tier: the pot's axis within 10 degrees of vertical
LOW_FOOT = 0.020  # the 0.25 tier: the pot's foot centre no more than 20 mm over the table top (on the tray or saucer)
TIP_DEG = 30.0  # any part's axis more than this from vertical at any moment: 0
OFF_DROP = 0.010  # a part's lowest point this far below the table top: it fell off the sill
NB_MOVE = 0.003  # the neighbour's base moved this far from its mark at any moment: disturbed
NB_TILT_DEG = 3.0  # or tilted this much: disturbed
LET_GO = 0.030  # both grippers' finger zones this far from the pot, the saucer and the neighbour
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # each part moves under 5 mm over the window (visible motion; the GPU demo ran at 0.0 mm)
STILL_TURN_DEG = 5.0  # and turns under 5 degrees
EPS = 1e-9
CONTROL_HZ = 20.0
START_TOL = 0.003  # the parts within 3 mm of their marks: the starting layout
RESET_JUMP = 0.08  # a part that lands on its mark having moved this much in one control step: a new episode began

FRICTION = (0.5, 0.4)  # plastic on plastic (est); the window is left at the simulator's default

# -- spawn poses (the base frame of each loose part is the centre of its underside) -----------------------------------
SAU_ORIGIN = (BAY_B[0], BAY_B[1], TOP + TRAY_FLOOR)
NB_ORIGIN = (BAY_A[0], BAY_A[1], TOP + TRAY_FLOOR)
POT_ORIGIN = (POT_X0, POT_Y0, TOP + DRAIN_FLOOR)
POT_SEATED_Z = TOP + TRAY_FLOOR + SAU_FLOOR  # the pot's foot on the well floor
PAD_TIP = 0.013  # the pad tips are 13 mm below the TCP


def corners(width_x, width_y, z=0.0):
    """The four corners (counter-clockwise from +x +y) of a rectangle centred on the frame's origin."""
    return tuple((sx * width_x / 2, sy * width_y / 2, z) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)))


def foot_corners():
    return corners(POT_BASE, POT_BASE)


def base_corners():
    return corners(SAU_BASE, SAU_BASE)


def yaw_fold_deg(d):
    """A turn in degrees folded into (-45, 45]: a square looks the same every quarter turn."""
    return (d + 45.0) % 90.0 - 45.0 if math.isfinite(d) else math.nan
