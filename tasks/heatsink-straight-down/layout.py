"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top

# The circuit board: a thin green slab fixed to the table, long side along x.
BOARD_X, BOARD_Y = 0.30, 0.0
BOARD_SIZE = (0.200, 0.160, 0.004)
BOARD_TOP = TOP + BOARD_SIZE[2]

# The processor die: a fixed 30 x 30 x 3 mm block centred on the board's top.
DIE_X, DIE_Y = BOARD_X, BOARD_Y
DIE_SIZE = (0.030, 0.030, 0.003)
DIE_HALF = DIE_SIZE[0] / 2
DIE_TOP = BOARD_TOP + DIE_SIZE[2]  # TOP + 0.007

# Four retention posts: fixed upright cylinders on the board at the corners of a 68 mm square around the die.
# A centred heatsink clears each post by 2 mm along x and y (4.5 mm corner to post).
POST_RADIUS = 0.004
POST_HEIGHT = 0.014
POST_OFFSET = 0.034  # from the die centre, along x and along y
POST_TOP = BOARD_TOP + POST_HEIGHT  # 11 mm above the die top
POSTS = (  # (name, x, y)
    ("post_front_left", BOARD_X + POST_OFFSET, BOARD_Y + POST_OFFSET),
    ("post_front_right", BOARD_X + POST_OFFSET, BOARD_Y - POST_OFFSET),
    ("post_back_left", BOARD_X - POST_OFFSET, BOARD_Y + POST_OFFSET),
    ("post_back_right", BOARD_X - POST_OFFSET, BOARD_Y - POST_OFFSET),
)

# The heatsink: a loose aluminium block standing on the table to the left of the board.
HS_SIZE = (0.056, 0.056, 0.032)
HS_HALF = HS_SIZE[0] / 2
HS_MASS = 0.20
HS_X, HS_Y = 0.22, 0.22

# Rule thresholds (scoring.py).
TOUCH_TOL = 0.002  # 'touching': the heatsink's bottom within 2 mm of the die's top, footprint over the die
FLAT_DEG = 2.0  # 'flat': tilted less than this
TIP_DEG = 30.0  # tilted more than this at the end: 0
CENTRE_TIGHT = 0.002  # centre to the die centre, horizontally (1.0 and 0.75)
CENTRE_LOOSE = 0.005  # (0.5)
YAW_TIGHT_DEG = 3.0  # sides to the board's axes (1.0 and 0.75)
YAW_LOOSE_DEG = 10.0  # (0.5)
SMEAR_TIGHT = 0.001  # smear for 1.0
SMEAR_LOOSE = 0.005  # smear for 0.75
SMEAR_TURN_RADIUS = DIE_HALF * math.sqrt(2.0)  # 21.2 mm: a turn on the die counts as this radius times the angle
# Smear is followed in steps of 0.2 mm: a move counts once the heatsink is this far (centre travel plus turn) from
# where smear was last counted, so a shiver that stays within 0.2 mm (contact jitter) does not add up over many
# steps; whatever is left is counted when it stops touching.
SMEAR_STEP = 0.0002
SETDOWN_WINDOW = 0.010  # the set-down speed is measured over the last 10 mm before it comes to touch
SETDOWN_MAX = 0.10  # m/s
POST_TOL = 0.002  # 'on a post': the heatsink's underside within 2 mm of a post's top, over it
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the heatsink moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
PCB_GREEN = (0.10, 0.40, 0.20)
DIE_GREY = (0.20, 0.20, 0.22)
POST_BLACK = (0.05, 0.05, 0.05)
ALUMINIUM = (0.80, 0.81, 0.83)
