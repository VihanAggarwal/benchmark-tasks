"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The robot cell: a fixed U of three yellow walls (a safety-fence stand-in) on the table, open at the front (the
# side toward the arm bases, x 0.25) and at the top. The table inside the walls is the cell's floor.
WALL_THICK = 0.010
WALL_HEIGHT = 0.060
BACK_WALL_X = 0.365  # centre of the wall that runs along y at the back of the cell (its faces at x 0.36 and 0.37)
BACK_WALL_LEN = 0.200  # it spans y -0.10..+0.10
SIDE_WALL_Y = 0.095  # centres of the two walls that run along x (their inner faces at y +-0.09)
SIDE_WALL_X0, SIDE_WALL_X1 = 0.255, 0.375  # they span this x range (overlapping the back wall's ends)
SIDE_WALL_X = (SIDE_WALL_X0 + SIDE_WALL_X1) / 2  # 0.315
SIDE_WALL_LEN = SIDE_WALL_X1 - SIDE_WALL_X0  # 0.120
# The cell's floor and the footprint of the 'safeguarded volume' the scorer watches: x 0.26..0.36, y -0.09..+0.09,
# from the table top up to CELL_TOP_Z. A gripper whose finger zone is wholly above CELL_TOP_Z is outside it.
CELL_X0, CELL_X1 = 0.260, 0.360
CELL_Y0, CELL_Y1 = -0.090, 0.090
CELL_HEIGHT = 0.300
CELL_TOP_Z = TOP + CELL_HEIGHT
# The front boundary of the cell is marked by a thin fixed red stripe plate on the table, centred on x 0.255:
# x >= 0.26 (its far edge) is inside the cell.
STRIPE_X = 0.255
STRIPE_SIZE = (0.010, 0.180, 0.002)  # along x, along y, thickness

# The dropped part: a loose blue aluminium block lying on the cell floor, sides parallel to the axes.
PART_SIZE = (0.030, 0.030, 0.030)
PART_MASS = 0.070  # 27 cm3 of aluminium is 73 g; a lightly chamfered block
PART_X, PART_Y = 0.310, 0.020
PART_HALF = PART_SIZE[0] / 2

# The enabling pad: a fixed yellow cylinder (a hold-to-run button) standing on the table on the left side, out of
# the right arm's reach (the right arm reaches no further left than about y +0.17 at this x).
PAD_X, PAD_Y = 0.200, 0.240
PAD_RADIUS = 0.020
PAD_HEIGHT = 0.020
PAD_TOP_Z = TOP + PAD_HEIGHT

# The recovery tray: four fixed grey walls round a floor 80 mm along x and 140 mm along y (the table top), on the
# right side, out of the left arm's reach (the left arm reaches no further right than about y -0.18 at this x).
# 140 mm along y, the fingers' closing axis at yaw 0, leaves the open fingers (94 mm apart plus two 8 mm pads)
# 15 mm to each end wall.
TRAY_X, TRAY_Y = 0.200, -0.220
TRAY_INNER_X, TRAY_INNER_Y = 0.080, 0.140
TRAY_WALL = 0.006
TRAY_HEIGHT = 0.015
TRAY_TOP_Z = TOP + TRAY_HEIGHT

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper (13 mm past the TCP to 25 mm
#                     behind it; the flat pad faces are about 22 mm of that)

# Rule thresholds (scoring.py).
PRESS_XY = 0.016  # 'pressed': a gripper's tip point within this of the pad's axis horizontally,
PRESS_Z_BELOW = 0.005  # between this far below the pad's top
PRESS_Z_ABOVE = 0.010  # and this far above it,
PRESS_OPENING = 0.5  # with the fingers at most this open (47 mm apart)
RELEASE_Z = 0.030  # 'released': the tip point more than this above the pad's top,
RELEASE_XY = 0.030  # or more than this from its axis horizontally (clearly off the button, not hovering on it)
LIFT_CLEAR = 0.020  # 'lifted': the part's lowest point at least this far above the table
IN_TRAY_Z_TOL = 0.003  # 'in the tray': the part's lowest point within this of the table top, its centre over the floor
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone at least this far from the part: it let go
OFF_TABLE_DROP = 0.010  # a part whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)
START_Z_TOL = 0.003  # and its lowest point within this of the table top

# Colours (sRGB).
FENCE_YELLOW = (0.95, 0.80, 0.10)
PAD_YELLOW = (1.00, 0.90, 0.20)
STRIPE_RED = (0.85, 0.10, 0.10)
PART_BLUE = (0.15, 0.35, 0.85)
TRAY_GREY = (0.55, 0.55, 0.58)
