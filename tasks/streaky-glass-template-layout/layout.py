"""Every dimension, position, mass, colour and threshold of the task, in one place (metres, kilograms, radians unless
named otherwise; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the glass sheet -------------------------------------------------------------------------------------------
# A streaky amber cathedral glass stand-in, taped down (fixed): 180 mm (x) by 300 mm (y) by 3 mm, built from 12 fixed
# strips 25 mm wide along y, each running the sheet's full 180 mm along x. Alternate strips are dark and light amber,
# so the streaks run front to back (along x), across the sheet's short side. 25 mm is about 4 to 5 pixels of the
# 320 x 176 top camera (1.7 to 2.0 px/cm), so each streak resolves and every template covers at least one boundary.
SHEET_X0, SHEET_X1 = 0.200, 0.380  # near and far long edges
SHEET_Y0, SHEET_Y1 = -0.150, 0.150  # the two short ends
SHEET_X = (SHEET_X0 + SHEET_X1) / 2  # 0.290
SHEET_Y = (SHEET_Y0 + SHEET_Y1) / 2  # 0.000
SHEET_THICK = 0.003
SHEET_TOP = TOP + SHEET_THICK
N_STRIPS = 12
STRIP_W = 0.025
STRIP_SIZE = (SHEET_X1 - SHEET_X0, STRIP_W, SHEET_THICK)  # 180 x 25 x 3 mm
STRIP_Y = tuple(SHEET_Y0 + STRIP_W / 2 + STRIP_W * k for k in range(N_STRIPS))  # -0.1375 + 0.025 k
DARK_AMBER = (0.62, 0.32, 0.06)  # even k
LIGHT_AMBER = (0.96, 0.76, 0.38)  # odd k

# -- the templates ---------------------------------------------------------------------------------------------
# White acrylic blocks standing in for paper pattern templates. Each box's size is (length, width, thickness): its
# body x axis is its long side, which carries the grain arrow (stated in the setup, not modelled). 12 mm thick so a
# top-down gripper can hold one by its ends with an 8 mm grip band.
A_SIZE = (0.070, 0.040, 0.012)
B_SIZE = (0.060, 0.040, 0.012)
A_MASS = 0.040  # 33.6 cm3 of acrylic (1.19 g/cm3)
B_MASS = 0.034  # 28.8 cm3
WHITE = (0.95, 0.95, 0.93)
# name: (size, mass, start x, start y, start yaw of the long side)
TEMPLATES = {
    "template_a1": (A_SIZE, A_MASS, 0.200, 0.360, math.pi / 2),  # long side along y: across the streaks
    "template_a2": (A_SIZE, A_MASS, 0.320, 0.360, 0.0),  # long side along x: already with the streaks
    "template_b": (B_SIZE, B_MASS, 0.260, -0.360, -math.pi / 4),  # turned 45 deg, far end toward -y
}
NAMES = tuple(TEMPLATES)

# The nominal end state (the demo's targets): one row along y near the near long edge, long sides along x.
ROW_X = 0.254
PLACE_Y = {"template_a1": 0.060, "template_a2": 0.000, "template_b": -0.060}

# -- the rules (scoring.py) ------------------------------------------------------------------------------------
TILT_MAX_DEG = 3.0  # (a) flat: tilted less than this
BOTTOM_TOL = 0.002  # (b) the template's lowest point within this of the sheet's top
STILL_WINDOW_S = 0.5  # (c) at rest: the centre's path over this window ...
STILL_PATH = 0.002  # ... is under this (a path of positions, never a velocity)
LETGO_DIST = 0.030  # (d) both grippers' finger zones (below) at least this far from the template's box
EDGE_MIN = 0.008  # (e) every corner of its outline seen from above at least this far inside every sheet edge
GAP_MIN = 0.005  # (f) at least this far from every other template seen from above ...
GAP_BAND = 0.020  # ... that lies low: its lowest point within this of the sheet's top (on the sheet, on another
#                   template or on the table); a template carried higher than this is not counted
GRAIN_MAX_DEG = 5.0  # (g) the long side within this of the streaks (x), either way round
STRIP_MIN = 0.080  # 1.0: a clear strip at least this wide along one long edge (inclusive)
SCORE_PER_LAID = (0.0, 0.2, 0.4, 0.6)  # 0..3 templates laid
EPS = 1e-9  # inclusive comparisons against the stated limits

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open) and
# the finger-tip point (midway between the two pad tips, 13 mm past the TCP). The finger zone covers the lowest 38 mm
# of each finger (yam_arena's finger collision meshes: 6.4 mm thick and 27 mm wide at the tips, 10 mm thick and 40 mm
# wide 19-38 mm up) as a round zone about the gripper's axis, because the scorer cannot see the gripper's heading:
# radius hypot(half the gap + 10 mm, 20 mm), 60 mm fully open; its flat bottom is the finger-tip point's plane.
FULL_GAP = 0.0939
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
FINGER_LEN = 0.038

# Episode reset: a template moving this far in one control step was teleported; a new episode began when every
# template is then back on its start mark (within START_TOL of it, flat on the table).
RESET_JUMP = 0.08
START_TOL = 0.045
