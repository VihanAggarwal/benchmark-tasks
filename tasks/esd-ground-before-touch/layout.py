"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The ESD mat: a fixed grey static-dissipative plate on the midline, the only place a board may be set down.
MAT_X, MAT_Y = 0.30, 0.0
MAT_SIZE = (0.220, 0.200, 0.003)  # 220 mm along x, 200 mm along y
MAT_HALF_X, MAT_HALF_Y = MAT_SIZE[0] / 2, MAT_SIZE[1] / 2
MAT_TOP = TOP + MAT_SIZE[2]

# The ground stud: a fixed gold post behind the mat, on the midline, which both arms reach.
STUD_X, STUD_Y = 0.14, 0.0
STUD_RADIUS = 0.010
STUD_HEIGHT = 0.060
STUD_TOP = TOP + STUD_HEIGHT

# The two shielding boxes: four fixed silver walls round a 140 x 140 mm floor (the table top), one in front of each
# arm, far enough forward that a cup on the parking mark is 30 cm from a board lifted out of either box. 140 mm
# across the fingers' closing axis (y) leaves the open fingers (94 mm apart plus 8 mm pads) 15 mm to each side
# wall; 140 mm along x leaves 10 mm at each end of a 120 mm board.
BOX_X = 0.41
BOX_Y = {"left": 0.32, "right": -0.32}
BOX_INNER = 0.140
BOX_INNER_HALF = BOX_INNER / 2
BOX_WALL = 0.006
BOX_HEIGHT = 0.025
BOX_RIM_Z = TOP + BOX_HEIGHT

# The boards: a circuit board in its carrier, one lying flat and centred in each box, long side along x.
BOARD_SIZE = (0.120, 0.060, 0.020)
BOARD_MASS = 0.080  # a 120 x 60 mm board in a conductive plastic carrier frame (mostly air inside: 0.56 g/cm3)
BOARD_HALF_X, BOARD_HALF_Y = BOARD_SIZE[0] / 2, BOARD_SIZE[1] / 2
BOARD_NAMES = {"left": "left_board", "right": "right_board"}

# The foam cup: a loose white lidded expanded-polystyrene cup standing on the mat, in the way of the right board.
CUP_X, CUP_Y = 0.30, -0.05
CUP_RADIUS = 0.025
CUP_HEIGHT = 0.070
CUP_MASS = 0.030  # a 4-oz foam cup with its lid and a little cold coffee left in it

# The parking square: a fixed yellow mark in the back right corner of the table, outside the work area. A cup
# standing on it is 33 cm from a board lifted out of the right box and 48 cm from the mat.
PARK_X, PARK_Y = 0.08, -0.60
PARK_SIZE = (0.080, 0.080, 0.002)
PARK_TOP = TOP + PARK_SIZE[2]
PARK_REST_TOL = 0.003  # a board's lowest point this close to the square's top: resting on it (bench, not allowed)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Rule thresholds (scoring.py).
TOUCH_GAP = 0.005  # a gripper's finger zone this close to a board's (or the cup's) box: touching it
STUD_AXIS_TOL = 0.015  # grounding: the TCP this close (horizontally) to the stud's axis ...
STUD_DEPTH = 0.005  # ... the finger tips at least this far below the stud's top (and above the table) ...
STUD_CLOSED = 0.40  # ... and the gripper closed to this opening or less (37.6 mm; on the 20 mm stud about 0.21)
CUP_DISTANCE = 0.30  # the cup's outline at least this far (horizontally) from every board out of its box
BARE_BENCH_TOL = 0.001  # a board corner this close to the table top (or below it): touching the bare bench
BOX_FLOOR_SLACK = 0.002  # a corner within this of a box's inner walls still counts as over its floor
DROP_DEG = 30.0  # a board or the cup tilted more than this: dropped, knocked over
FLAT_DEG = 5.0  # 'flat': tilted less than this (either 120 x 60 mm face down)
ON_MAT_Z_TOL = 0.003  # 'on the mat': a board's lowest point within this of the mat's top
OVERHANG_TIGHT = 0.0  # 'fully on the mat': no part of the board's outline past the mat's edge (1.0)
OVERHANG_LOOSE = 0.015  # ... or past it by at most this (0.75)
BOARD_GAP = 0.020  # the two boards' outlines at least this far apart (1.0)
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone at least this far from both boards: it let go
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
GREEN = (0.10, 0.45, 0.20)  # circuit board solder mask
SILVER = (0.78, 0.79, 0.81)  # metallised shielding box
MAT_GREY = (0.45, 0.47, 0.50)
GOLD = (0.85, 0.68, 0.20)
WHITE = (0.97, 0.97, 0.97)
YELLOW = (0.95, 0.82, 0.10)
