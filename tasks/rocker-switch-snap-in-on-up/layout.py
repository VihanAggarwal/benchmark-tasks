"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A steel control panel (a 1.5 mm plate with one rectangular cutout) lies flat on two steel rails, as it does on an
assembler's bench, so the switch's body can pass through it. Its far edge (+x) is the panel's TOP edge and carries an
orange band. A KCD1-type rocker switch stands on a foam pad, rocker up, with its printed I half toward the robot
(toward the panel's bottom edge): wrong way round. It must be picked up, turned half a turn about the vertical and
pressed into the cutout until its flange lies on the plate, the I half toward the top edge.

The switch's own frame (``SWITCH`` below): origin at the centre of the flange's rear face (the face that lands on the
plate), z along the switch's axis toward the rocker (up when mounted), x toward its I half, y along the rocker's long
side. The body lies at z -25 mm .. 0, the flange 0 .. 1.5 mm, the rocker 1.5 .. 9.5 mm.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the panel (fixed) ---------------------------------------------------------------------------------------------
PX, PY = 0.36, 0.22  # the cutout's centre (world x, y); the plate is centred on it
PL_X, PL_Y, PL_T = 0.100, 0.160, 0.0015  # plate: 100 (x) x 160 (y) x 1.5 mm, grey steel
RAIL_X, RAIL_H = 0.015, 0.040  # two steel rails 15 mm wide (x), 160 long (y), 40 mm high, under the plate's edges
PLATE_TOP = RAIL_H + PL_T  # 41.5 mm above the table
PLATE_BOTTOM = RAIL_H
HOLE_X, HOLE_Y = 0.022, 0.030  # the cutout: 22.0 (x) x 30.0 (y) mm (KCD1-101 class mounting hole), square corners
BAND_X0, BAND_X1 = 0.036, 0.046  # the orange band on the far (top) edge: x offsets from the plate's centre
BAND_T = 0.0004
STEEL = (0.55, 0.57, 0.60)
RAIL_STEEL = (0.35, 0.36, 0.38)
ORANGE = (0.95, 0.45, 0.05)
PANEL_FRICTION = (0.45, 0.40)  # steel (est)

# -- the foam pad (fixed) ------------------------------------------------------------------------------------------
FOAM_X, FOAM_Y = 0.20, 0.22  # centre
FOAM_SX, FOAM_SY, FOAM_H = 0.080, 0.060, 0.010
FOAM_GREY = (0.30, 0.38, 0.52)

# -- the switch (loose) --------------------------------------------------------------------------------------------
# Generic KCD1-101 class rocker (20 A 250 V): body about 28 x 20.5 mm, flange 31 x 25, rocker proud of the flange. All
# sizes are catalogue-class estimates (est): the README says to measure the real part. 12 g.
BODY_Y, BODY_X, DEPTH = 0.028, 0.0205, 0.025  # body 28.0 (y) x 20.5 (x), 25 mm deep behind the flange
CHAMFER = 0.002  # the body's rear edges are eased 2 mm at 45 degrees (a lead-in; est)
FL_Y, FL_X, FL_T = 0.031, 0.025, 0.0015  # flange 31 (y) x 25 (x) x 1.5 mm
RK_Y, RK_X, RK_H = 0.025, 0.018, 0.008  # rocker 25 (y) x 18 (x), 8 mm proud of the flange
MARK_T = 0.0004
SW_MASS = 0.012
SW_BLACK = (0.06, 0.06, 0.07)
SW_WHITE = (0.96, 0.96, 0.94)
SW_FLANGE = (0.12, 0.12, 0.13)
SW_FRICTION = (0.40, 0.35)  # ABS on steel / on the gripper pad (est)
TOP_OF_ROCKER = FL_T + RK_H  # 9.5 mm above the flange's rear face
SW_YAW0 = math.pi  # as found: the I half toward the robot (-x)
SW_X0, SW_Y0 = FOAM_X, FOAM_Y
SW_Z0 = FOAM_H + DEPTH  # origin height above the table standing on the foam: 35 mm
SEAT_Z = PLATE_TOP  # the origin's height above the table when seated: the flange's rear face on the plate

# -- fit and tolerances (the physics decides; these are what the checker and the README quote) -----------------------
CLEAR_X = (HOLE_X - BODY_X) / 2  # 0.75 mm each side along x
CLEAR_Y = (HOLE_Y - BODY_Y) / 2  # 1.0 mm each side along y
CAPTURE_X = CLEAR_X + CHAMFER  # offline model only: offsets the chamfer can pull in
CAPTURE_Y = CLEAR_Y + CHAMFER
FIT_YAW_DEG = 3.0  # offline model only: a body turned more than this about the vertical jams on the cutout's edges

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SEAT_TOL = 0.0005  # in: the flange's rear face within 0.5 mm of the plate's top, at its highest corner
PROUD_MAX = 0.003  # 0.25: proud by up to 3 mm
SQUARE_DEG = 3.0  # square: the switch's axis within 3 degrees of the panel normal
CROOKED_DEG = 8.0  # 0.25: crooked up to 8 degrees
ORIENT_DEG = 20.0  # the I half within 20 degrees of the panel's top-edge direction (+x), or of the opposite
IN_HOLE_X, IN_HOLE_Y = 0.0025, 0.0030  # the origin within this of the cutout's centre to count as in it
FALL_Z = -0.010  # latched: the switch's lowest point more than 10 mm below the table top (it fell off the table)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.0005
STILL_TURN_DEG = 0.3
LET_GO = 0.030
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
