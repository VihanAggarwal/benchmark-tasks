"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top

# -- the jig's base plate: 300 x 200 x 6 mm, fixed, built from 7 boxes with flush tops (the lines are inlaid) -------
PLATE_X0, PLATE_X1 = 0.150, 0.450
PLATE_HALF_Y = 0.100
PLATE_T = 0.006
PLATE_TOP = TOP + PLATE_T  # 0.7535
LASER_X = 0.270  # the red laser line's centre: the reference line
LINE_W = 0.004  # both lines are 4 mm wide
_LH = LINE_W / 2
# (name, (x0, x1), (y0, y1), colour key)
PLATE_PIECES = (
    ("plate_rear_left", (PLATE_X0, LASER_X - _LH), (_LH, PLATE_HALF_Y), "grey"),
    ("plate_rear_right", (PLATE_X0, LASER_X - _LH), (-PLATE_HALF_Y, -_LH), "grey"),
    ("plate_front_left", (LASER_X + _LH, PLATE_X1), (_LH, PLATE_HALF_Y), "grey"),
    ("plate_front_right", (LASER_X + _LH, PLATE_X1), (-PLATE_HALF_Y, -_LH), "grey"),
    ("laser_line", (LASER_X - _LH, LASER_X + _LH), (-PLATE_HALF_Y, PLATE_HALF_Y), "red"),
    ("centre_line_rear", (PLATE_X0, LASER_X - _LH), (-_LH, _LH), "dark"),
    ("centre_line_front", (LASER_X + _LH, PLATE_X1), (-_LH, _LH), "dark"),
)

# -- the heel block: a loose polyurethane block, 60 (x) x 70 (y) x 20 mm (the shoe's 15 mm effective heel height plus
# 5 mm; the effective heel height is the heel's height minus the forefoot sole's thickness) ---------------------------
BLOCK_SIZE = (0.060, 0.070, 0.020)
BLOCK_HALF = tuple(s / 2 for s in BLOCK_SIZE)
BLOCK_MASS = 0.050
BLOCK_START = (0.250, 0.400)  # centre, standing flat on the table, yaw 0

# -- the right prosthetic foot: one loose mechkit body of three boxes -------------------------------------------------
# Its base frame: origin at the heel end of the sole, mid-width, sole level; +x toward the toe, +y to its left, +z up.
FOOT_LEN = 0.220
FOOT_W = 0.070
FOOT_MASS = 0.25
REAR_CENTER, REAR_SIZE = (0.060, 0.0, 0.014), (0.120, 0.070, 0.028)  # rear shell (heel and midfoot), beige
FORE_CENTER, FORE_SIZE = (0.170, 0.0, 0.008), (0.100, 0.070, 0.016)  # forefoot shell, beige; soles flush
ADAPTER_CENTER, ADAPTER_SIZE = (0.085, 0.0, 0.043), (0.046, 0.046, 0.030)  # pyramid adapter stand-in, black
FOOT_SHAPES = ((REAR_CENTER, REAR_SIZE), (FORE_CENTER, FORE_SIZE), (ADAPTER_CENTER, ADAPTER_SIZE))
FOOT_TOP = ADAPTER_CENTER[2] + ADAPTER_SIZE[2] / 2  # 0.058: the adapter's top above the sole
# Mass spread evenly over the volume (how PhysX does it): centre of mass 93.9 mm from the heel end, 16.8 mm up.
FOOT_COM_S = 0.0939
FOOT_ORIGIN = (0.190, -0.380, TOP)  # start: heel end at x 0.190, toe end at x 0.410, toes along +x, sole on the table
# Offline only (tools/offline/check_task.py): the checker draws and grips the foot as this column, the adapter's
# 46 x 46 mm outline from the sole up to the adapter's top (see task.py, Foot).
COLUMN_CENTER = (ADAPTER_CENTER[0], 0.0, FOOT_TOP / 2)
COLUMN_SIZE = (ADAPTER_SIZE[0], ADAPTER_SIZE[1], FOOT_TOP)

# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------
# The gate (all needed, else 0):
FOOT_TILT_MAX_DEG = 15.0  # (G2) the foot's up axis within this of the vertical (sole down)
HEADING_MAX_DEG = 20.0  # (G3) heel-to-toe, seen from above, within this of +x (toes forward). 20, not 30: with G1
#                         and G4 a foot turned past about 22 deg already has a corner off the plate
HEEL_SIDE_MAX = 0.015  # (G4) the heel point within this of the centre line (y 0), sideways
RELEASE_CLEAR = 0.020  # (G5) each arm's TCP and pad-tip point at least this far from every box of the foot and block
STILL_WINDOW_S = 0.5  # (G6) still: over the last 0.5 s ...
STILL_PATH = 0.002  # ... the heel point, the toe point and the block's centre each moved (path length) under this
# The three checks, 0.25 each on top of 0.25 for the gate:
AP_TARGET = 0.030  # (A) the foot's middle this far in front of the laser line's centre ...
AP_TOL = 0.005  # ... within this: 25 to 35 mm
TOE_OUT_MIN_DEG = 2.0  # (B) heel-to-toe turned from +x toward -y (the robot's right) by 2 ...
TOE_OUT_MAX_DEG = 8.0  # ... to 8 degrees
BLOCK_TILT_MAX_DEG = 10.0  # (C) the block flat: its 20 mm axis within this of the vertical (either face down) ...
BLOCK_BOTTOM_TOL = 0.003  # ... and its lowest point within this of the plate's top
BLOCK_SIDE_MAX = 0.035  # (C) the block's centre within this of the foot's long axis, sideways ...
BLOCK_ALONG_MIN = -0.030  # ... and no more than 30 mm behind the heel point along it (toward the toe +) ...
BLOCK_FRONT_MAX = 0.090  # ... and none of its corners, seen from above, more than this in front of the heel point
#                          along it: 4 mm behind the centre of mass (FOOT_COM_S, 93.9 mm), so a foot resting heel on
#                          the block and toe on the plate stays so by itself (for a block square to the foot this is
#                          its centre at most 60 mm in front of H; a block turned on the plate must sit further back)
HEEL_H_MIN, HEEL_H_MAX = 0.018, 0.040  # (C) the heel point this high above the plate's top
TOE_T_TOL = 0.004  # (C) the toe point within this of the plate's top

CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported ...
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)
START_Z_TOL = 0.003
START_TILT_DEG = 3.0

# The grippers (yam_arena): the TCP is 13 mm above the point between the pad tips.
PAD_TIPS = 0.013

# Colours (sRGB).
PLATE_GREY = (0.75, 0.76, 0.78)
LASER_RED = (0.90, 0.10, 0.10)
LINE_DARK = (0.20, 0.20, 0.22)
BLOCK_ORANGE = (0.95, 0.50, 0.10)
FOOT_BEIGE = (0.85, 0.72, 0.58)
ADAPTER_BLACK = (0.08, 0.08, 0.08)
PIECE_COLOURS = {"grey": PLATE_GREY, "red": LASER_RED, "dark": LINE_DARK}

TOE_OUT_RULER = (math.tan(math.radians(TOE_OUT_MIN_DEG)), math.tan(math.radians(TOE_OUT_MAX_DEG)))
