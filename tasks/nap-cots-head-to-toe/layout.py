"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. The scene is a nap room at 1:8: the table top is its floor.
"""

TOP = 0.7475  # station table top (the room's floor)

# The rest-area rug: a thin fixed slab; it is "the room" the cots must stay inside.
RUG_X, RUG_Y = 0.280, 0.0
RUG_SIZE = (0.240, 0.480, 0.002)  # x 0.160..0.400, y -0.240..+0.240
RUG_X0, RUG_X1 = RUG_X - RUG_SIZE[0] / 2, RUG_X + RUG_SIZE[0] / 2
RUG_Y0, RUG_Y1 = RUG_Y - RUG_SIZE[1] / 2, RUG_Y + RUG_SIZE[1] / 2
RUG_TOP = TOP + RUG_SIZE[2]

# Three nap cots: loose solid boxes, long side along x, standing side by side on the rug.
COT_SIZE = (0.165, 0.069, 0.020)
COT_HALF_LEN = COT_SIZE[0] / 2
COT_MASS = 0.060
COT_TOP = RUG_TOP + COT_SIZE[2]  # the cots' top face, set down on the rug: 22 mm above the floor
COTS = ("cot_left", "cot_middle", "cot_right")
COT_START = {  # centres (x, y): 51 mm of floor between neighbours
    "cot_left": (0.2775, 0.120),
    "cot_middle": (0.2775, 0.000),
    "cot_right": (0.2775, -0.120),
}

# Three pillows: loose rigid blocks (white-painted hardwood in a real build), one on each cot, all at the cots' far ends.
PILLOW_SIZE = (0.030, 0.044, 0.012)  # 30 mm along the cot, 44 mm across it, 12 mm thick
PILLOW_MASS = 0.012
PILLOWS = ("pillow_left", "pillow_middle", "pillow_right")
END_INSET = 0.025  # a pillow at an end of a cot: its centre this far in from that end
PILLOW_START = {  # centres (x, y): 25 mm in from each cot's far end (x 0.360)
    "pillow_left": (0.2775 + COT_HALF_LEN - END_INSET, 0.120),
    "pillow_middle": (0.2775 + COT_HALF_LEN - END_INSET, 0.000),
    "pillow_right": (0.2775 + COT_HALF_LEN - END_INSET, -0.120),
}
START = {**COT_START, **PILLOW_START}

# Rule thresholds (scoring.py).
GAP_MIN = 0.114  # 3 ft (914 mm) at 1:8: the clear floor between neighbouring cots, seen from above
END_BAND = 0.050  # a pillow marks a cot's end when its centre is within this of that end
FLAT_TOL = 0.003  # a cot set down: its lowest point within this of the rug's top
TILT_MAX_DEG = 5.0  # ... its broad faces within this of horizontal
HEADING_MAX_DEG = 15.0  # ... its long side within this of x (pointing forward, as it started)
ON_COT_TOL = 0.003  # a pillow on a cot: its lowest point within this of the cot's top
REST_WINDOW_S = 0.5  # 'at rest': no tracked point moved more than REST_MOVE in this long
REST_MOVE = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.040  # an object moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... back onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
RUG_GREEN = (0.30, 0.55, 0.30)
COT_BLUE = (0.15, 0.35, 0.80)
PILLOW_WHITE = (0.95, 0.95, 0.93)
