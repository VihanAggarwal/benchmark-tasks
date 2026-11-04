"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The sterile jar: a square open-top jar clamped to the table (fixed), its floor the table top.
JAR_X, JAR_Y = 0.26, 0.0
JAR_OUTER = 0.066  # outer width, both ways
JAR_WALL = 0.008
JAR_HEIGHT = 0.070
JAR_INNER_HALF = JAR_OUTER / 2 - JAR_WALL  # 0.025: the opening is 50 x 50 mm
RIM_Z = TOP + JAR_HEIGHT

# The lid: a flat square lid that rests on the rim, overhanging it by 4 mm all round.
LID_SIZE = (0.074, 0.074, 0.018)
LID_MASS = 0.040

# The sample cube, on the table to the right of the jar.
CUBE_EDGE = 0.024
CUBE_MASS = 0.012
CUBE_X, CUBE_Y = 0.18, -0.24

# Rule thresholds (scoring.py).
SEAT_XY_TOL = 0.005  # lid centre to jar axis, horizontally
SEAT_Z_TOL = 0.004  # lid bottom to rim top
SEAT_TILT_DEG = 5.0
BENCH_CLEARANCE = 0.030  # while off the jar, the lid's lowest point must stay this high above the table top
LIFT_CLEAR = 0.030  # 'lifted': the lid's lowest point this far above the rim
OPEN_LIMIT_S = 20.0  # the jar may be open (lid not seated) this long in total
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # an object moving this far in one control step was teleported: a new episode began

# Colours (sRGB).
WHITE = (0.93, 0.93, 0.93)
BLUE = (0.15, 0.35, 0.85)
YELLOW = (0.95, 0.80, 0.10)
