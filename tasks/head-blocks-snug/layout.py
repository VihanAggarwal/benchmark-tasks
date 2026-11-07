"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The spine board: a fixed yellow slab along x, its head end toward the arms.
BOARD_X0, BOARD_X1 = 0.120, 0.600
BOARD_HALF_W = 0.150  # 300 mm wide, centred on y 0
BOARD_T = 0.015
BOARD_TOP = TOP + BOARD_T

# The casualty trainer lies on its back on the board, head toward the arms. Torso and neck are fixed (strapped);
# the head is a separate weighted part resting on the board, so anything that pushes it moves it.
TORSO_X0, TORSO_X1 = 0.300, 0.600
TORSO_HALF_W = 0.100
TORSO_H = 0.050
NECK_X0, NECK_X1 = 0.270, 0.300
NECK_HALF_W = 0.030
NECK_H = 0.030
HEAD_X, HEAD_Y = 0.200, 0.0  # the head's start: centred on the board's centre line, 10 mm short of the neck
HEAD_R = 0.060  # 120 mm across
HEAD_H = 0.100  # 100 mm tall (back of the head to the face)
HEAD_MASS = 0.30
HEAD_Z = BOARD_TOP + HEAD_H / 2

# Two foam head blocks, one in each arm's column, standing on the table beside the board, 70 mm side along x.
BLOCK_SIZE = (0.070, 0.050, 0.080)  # x (along the board), y (across), z (tall)
BLOCK_HALF = tuple(s / 2 for s in BLOCK_SIZE)
BLOCK_MASS = 0.040
BLOCK_X, BLOCK_Y = 0.200, 0.280
BLOCKS = {  # name -> start centre (x, y)
    "head_block_left": (BLOCK_X, BLOCK_Y),
    "head_block_right": (BLOCK_X, -BLOCK_Y),
}

# Rule thresholds (scoring.py).
SNUG_GAP = 0.005  # 0.5 for a side: a block beside the head with a gap of 5 mm or less
CLOSE_GAP = 0.015  # 0.25: 15 mm or less
SNUG_CREDIT, CLOSE_CREDIT = 0.5, 0.25
OVER_HEAD = 0.005  # any block reaching more than this inside the head's outline seen from above: 0
UPRIGHT_MAX_DEG = 10.0  # a block's 80 mm axis within this of vertical
HEADING_MAX_DEG = 15.0  # its 70 mm side within this of the board's long axis (x), seen from above
ON_BOARD = 0.003  # its lowest point within this of the board's top
CENTRED_X = 0.025  # its centre within this of the head's centre, along x
HEAD_SHIFT_MAX = 0.008  # the head's centre ever more than this from its start, seen from above: 0
HEAD_LIFT_MAX = 0.010  # the head's lowest point ever more than this above the board's top: 0
HEAD_TILT_MAX_DEG = 10.0  # the head's axis ever more than this from vertical: 0
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)
HEAD_START_TOL = 0.004  # the head counts as on its start mark within this (the scene's starting layout)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone more than this from a block: it has let go of it

# Colours (sRGB).
YELLOW = (0.95, 0.78, 0.12)
TAN = (0.80, 0.66, 0.50)
ORANGE = (0.96, 0.45, 0.08)
