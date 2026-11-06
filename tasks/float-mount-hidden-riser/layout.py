"""Every dimension, position, mass, colour and threshold of the task, in one place (metres, kilograms, radians
unless a name says otherwise; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left of their midpoint, z up. The arm
bases stand at x 0, y +-0.305; the station's table top is at z 0.7475. Heights in comments are above the table top.
"""

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# (1) The backing board: white acid-free foam board on a rigid base, fixed to the table, long side along x.
BOARD_X, BOARD_Y = 0.300, 0.000
BOARD_SIZE = (0.180, 0.130, 0.005)  # x 0.210..0.390, y -0.065..+0.065
BOARD_HALF_X, BOARD_HALF_Y = BOARD_SIZE[0] / 2, BOARD_SIZE[1] / 2
BOARD_TOP = TOP + BOARD_SIZE[2]  # 5 mm up

# (2) The riser: a black glued mat-board stack, loose, lying flat on the table on the right with its 70 mm length
# along y (body x is its 30 mm width, body y its 70 mm length, body z its 10 mm height).
RISER_SIZE = (0.030, 0.070, 0.010)
RISER_MASS = 0.015
RISER_X, RISER_Y = 0.240, -0.360  # x 0.225..0.255, y -0.395..-0.325
RISER_YAW = 0.0
RISER_TOP_ON_BOARD = BOARD_TOP + RISER_SIZE[2]  # 15 mm up

# (3) The print: dry-mounted on 10 mm rigid board, loose, lying face up (body +z) on the table on the left, its
# 110 mm long side (body x) along x. 22 g: 77 cm3 of 10 mm gatorboard (about 0.25 g/cm3) plus the print and the
# dry-mount tissue.
PRINT_SIZE = (0.110, 0.070, 0.010)
PRINT_HALF_X, PRINT_HALF_Y = PRINT_SIZE[0] / 2, PRINT_SIZE[1] / 2
PRINT_MASS = 0.022
PRINT_X, PRINT_Y = 0.240, 0.360  # x 0.185..0.295, y +0.325..+0.395

# The nominal end state: riser and print both centred on the board, both lengthwise along x; inset 20 mm all
# round (print 110 x 70 over riser 70 x 30), margins 35 mm along x and 30 mm along y.
NOMINAL_INSET = min(PRINT_SIZE[0] - RISER_SIZE[1], PRINT_SIZE[1] - RISER_SIZE[0]) / 2  # 0.020

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis)
FINGER_LEN = 0.038  # the finger zone runs from the tips this far back up the gripper

# Rule thresholds (scoring.py).
STILL_STEPS = 10  # 'at rest' is judged over the last 0.5 s (10 control steps) ...
STILL_PATH = 0.005  # ... the centre's path over that window shorter than this
STILL_TURN_DEG = 3.0  # ... and its turn (summed step by step) less than this
RISER_FLAT_DEG = 10.0  # the riser's 10 mm axis within this of vertical (either broad face down)
RISER_LOW_MIN, RISER_LOW_MAX = 0.003, 0.007  # the riser's lowest point this high above the table (board top 5 mm)
PRINT_FACE_UP_DEG = 3.0  # the print's face normal (body +z) within this of straight up
PRINT_LOW_MIN, PRINT_LOW_MAX = 0.013, 0.017  # the print's lowest point this high above the table (riser top 15 mm)
INSET_MIN = 0.010  # the riser set back at least this far from every edge of the print (0.75 and 1.0)
CENTRE_TIGHT = 0.006  # print centre to board centre, in plan (1.0)
CENTRE_LOOSE = 0.012  # (0.75)
TURN_TIGHT_DEG = 3.0  # print's long sides to the board's long sides, folded to 0..90 (1.0)
TURN_LOOSE_DEG = 8.0  # (0.75)
CLEAR_GAP = 0.030  # 'clear': both grippers' finger zones (flat-ended at the tips) this far or more from both parts
OFF_TABLE_DROP = 0.010  # a loose part whose lowest point is this far below the table top is off the table
RESET_JUMP = 0.08  # a part moving this far in one control step was put back by the simulator: a new episode ...
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
BOARD_WHITE = (0.95, 0.95, 0.93)
RISER_BLACK = (0.05, 0.05, 0.05)
PRINT_CREAM = (0.93, 0.89, 0.78)
