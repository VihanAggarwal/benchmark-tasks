"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The two bearers: fixed timber battens on the table, long side along y, the foundation of the stack.
BEARER_SIZE = (0.020, 0.320, 0.020)  # along x, along y, height
BEARER_XS = (0.20, 0.36)  # bearer_near, bearer_far; both centred at y 0 (they span y -0.16 to +0.16)
BEARER_Y = 0.0
BEARER_TOP = TOP + BEARER_SIZE[2]

# The four boards: loose red-cedar boards lying flat on the table, long side along x, two on each arm's side. Straight
# down an arm reaches only 0.10 m past the midline; tilted about 50 degrees, with its gripper a few mm off the
# table, it can just reach the back end (x 0.16-0.21) of the other side's inner board. The stickers decide it:
# each lies 0.85 m across the table from the other arm's base, out of its reach in any pose.
BOARD_SIZE = (0.240, 0.040, 0.016)  # along x, along y, thickness
BOARD_MASS = 0.060  # western red cedar, about 370-390 kg/m3
BOARD_X = 0.28  # in the stack: centred between the bearers along x, each board spans x 0.16 to 0.40
BOARD_START_X = 0.28  # at the start too: the boards are carried straight across y
BOARDS = (  # (name, y) at the start
    ("board_left_inner", 0.38),
    ("board_left_outer", 0.46),
    ("board_right_inner", -0.38),
    ("board_right_outer", -0.46),
)

# The two stickers: loose thin hardwood strips lying flat on the table outboard of each arm's boards. Their size is
# given along y, as they lie in the stack; at the start each is turned a quarter turn (STICKER_START_YAW), so it
# lies along x, spanning x 0.24 to 0.52.
STICKER_SIZE = (0.020, 0.280, 0.016)  # across, along, thickness (in the stack: along x, along y, up)
STICKER_MASS = 0.040  # stained poplar, about 450 kg/m3
STICKER_START_YAW = 1.5707963267948966  # pi/2: lying along x
STICKERS = (  # (name, x, y) at the start
    ("sticker_left", 0.38, 0.54),
    ("sticker_right", 0.38, -0.54),
)

# The intended stack (the demo builds this; the scorer accepts any board in any place and either sticker over
# either bearer): layer 1 two boards on the bearers centred at y +-0.05 (a 60 mm gap between them), a sticker on
# layer 1 over each bearer, layer 2 two boards on the stickers right above layer 1.
LAYER_Y = 0.05
LAYER1_BOTTOM = BEARER_TOP  # TOP + 0.020
LAYER1_TOP = LAYER1_BOTTOM + BOARD_SIZE[2]  # TOP + 0.036
STICKER_BOTTOM = LAYER1_TOP
STICKER_TOP = STICKER_BOTTOM + STICKER_SIZE[2]  # TOP + 0.052
LAYER2_BOTTOM = STICKER_TOP
LAYER2_TOP = LAYER2_BOTTOM + BOARD_SIZE[2]  # TOP + 0.068

# Rule thresholds (scoring.py).
REST_Z_TOL = 0.003  # 'rests on': the piece's underside within this of the support's top where they cross,
REST_COVER = 0.5  # and the crossing at least this share of (piece width x support width) seen from above
FLAT_DEG = 5.0  # 'flat': tilted less than this (either broad face up)
HEADING_DEG = 5.0  # 'along x' (boards) / 'along y' (stickers): long side within this of the axis, either way
EDGE_DEG = 45.0  # tilted more than this: standing on an edge or end (0)
OFF_TABLE_DROP = 0.010  # a piece's lowest point this far below the table top: it fell off the table (0)
GAP_MIN, GAP_MAX = 0.030, 0.090  # the gap between the two layer-1 boards (their centres' y distance minus 40 mm)
STICKER_TIGHT, STICKER_LOOSE = 0.010, 0.025  # a sticker's centre to its bearer's x (1.0 / 0.75)
STACK_TIGHT, STACK_LOOSE = 0.015, 0.030  # a layer-2 board's centre to its layer-1 board's centre, in y (1.0 / 0.75)
STILL_SPEED, STILL_SPIN = 0.01, 0.5  # 'still': under 1 cm/s and 0.5 rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a piece that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
WEATHERED = (0.55, 0.54, 0.50)  # silver-grey weathered timber: unlike the dark-brown stickers
CEDAR = (0.62, 0.30, 0.18)  # red-brown: stands out from the oak-coloured station table
DARK_STAIN = (0.18, 0.14, 0.12)  # near-black: distinct from the boards and the table
