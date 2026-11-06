"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Bearings are measured anticlockwise from +x, seen from
above, about the trap's centre.
"""

import math

TOP = 0.7475  # station table top

# The pitfall trap: a fixed black disc standing in for the open mouth of a bucket sunk flush with the ground.
# Its rim is the 35 mm radius.
TRAP_X, TRAP_Y = 0.270, 0.000
TRAP_RADIUS = 0.035
TRAP_HEIGHT = 0.003

# Four fixed orange pegs, each marking the far end of one fence arm: 200 mm from the trap's centre at bearings
# 45, 135, 225 and 315 deg (lines 1 to 4). Lines 1 and 3, and 2 and 4, are collinear through the trap's centre.
PEG_RADIUS = 0.006
PEG_HEIGHT = 0.060
PEG_DIST = 0.200
PEG_BEARINGS_DEG = (45.0, 135.0, 225.0, 315.0)
PEGS = tuple(
    (f"peg_{k + 1}", TRAP_X + PEG_DIST * math.cos(math.radians(b)), TRAP_Y + PEG_DIST * math.sin(math.radians(b)))
    for k, b in enumerate(PEG_BEARINGS_DEG)
)  # P1 (0.4114, +0.1414), P2 (0.1286, +0.1414), P3 (0.1286, -0.1414), P4 (0.4114, -0.1414)

# Four loose pine boards, 130 mm long (x), 20 mm thick (y), 50 mm tall (z), each standing on a 130 x 20 mm edge
# with its long side along x. Pine at about 500 kg/m^3: 65 g. The boards in a pair are 60 mm apart face to face.
BOARD_SIZE = (0.130, 0.020, 0.050)
BOARD_HALF = tuple(s / 2 for s in BOARD_SIZE)
BOARD_MASS = 0.065
BOARDS = ("board_1", "board_2", "board_3", "board_4")
BOARD_STARTS = {
    "board_1": (0.240, +0.370),
    "board_2": (0.240, +0.450),
    "board_3": (0.240, -0.370),
    "board_4": (0.240, -0.450),
}

# Rule thresholds (scoring.py).
STAND_TILT_DEG = 10.0  # 'standing': the board's 50 mm axis within this of vertical
STAND_LOW_TOL = 0.003  # ... and its lowest point within this of the table top
REST_STEPS = 10  # 'at rest': every centre recorded over the last 0.5 s (the current step and the 10 before) ...
REST_TOL = 0.002  # ... within this of the current one (positions only: a gripped part's velocity chatters)
# 'Free': no gripper TCP within FREE_RADIUS, in plan, of the board's footprint while that TCP is less than
# FREE_HEIGHT above the table. Fully open (94 mm), a pad's outer corner is 56 mm from the TCP in plan (47 mm half
# gap + 8 mm pad, 10 mm half width), so a pad touching a board puts the TCP within 60 mm of its footprint.
FREE_RADIUS = 0.060
FREE_HEIGHT = 0.090
AXIS_TOL_DEG = 8.0  # fenced: the long axis within this of the trap-to-peg direction (either end to the trap)
CENTRELINE_TOL = 0.010  # fenced: the trap's centre within this, in plan, of the board's long centreline
GAP_MIN = -0.001  # fenced: gap (plan distance from the trap's centre to the footprint, minus the radius) from ...
GAP_MAX = 0.010  # ... GAP_MIN to GAP_MAX
ZERO_GAP = -0.001  # any board with a gap below this at the scored step (on or over the opening): score 0
CREDIT_PER_LINE = 0.25

CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a board moving this far in one control step was teleported ...
START_TOL = 0.045  # ... and, with every board within this of its own starting mark, a new episode began

# Colours (sRGB).
TRAP_BLACK = (0.05, 0.05, 0.05)
PEG_ORANGE = (0.95, 0.45, 0.05)
PINE = (0.85, 0.72, 0.50)
