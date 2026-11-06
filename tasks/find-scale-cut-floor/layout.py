"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights in comments are above the table top.

The scene is a small excavation: a square frame of four 40 mm blocks (the ground surface) round a sunken cut whose
floor is the table top. A find lies on the cut floor; a trowel lies beside it, on the strip where the photo scale
belongs; an orange tool tray stands to the right (only the right arm reaches it) and a grey photo-kit box with the
scale in it to the left (only the left arm reaches it).
"""

import math

TOP = 0.7475  # station table top: the cut floor
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the ground surface: four fixed 40 mm blocks round the cut (outer 300 x 300 mm, centred x 0.26, y 0) ----------
BANK_H = 0.040
BANKS = (  # (name, centre x, centre y, size x, size y)
    ("near_bank", 0.14, 0.00, 0.060, 0.300),  # x 0.11-0.17, y -0.15..+0.15
    ("far_bank", 0.38, 0.00, 0.060, 0.300),  # x 0.35-0.41
    ("left_bank", 0.26, 0.12, 0.180, 0.060),  # y +0.09..+0.15, x 0.17-0.35
    ("right_bank", 0.26, -0.12, 0.180, 0.060),  # y -0.15..-0.09
)
FRAME = (0.11, 0.41, -0.15, 0.15)  # outer outline: x0, x1, y0, y1
CUT = (0.17, 0.35, -0.09, 0.09)  # the cut opening (180 x 180 mm); its floor is the table top
GROUND_Z = TOP + BANK_H  # the ground surface, 40 mm above the cut floor

# -- the find: a loose stand-in for a worked bone point, flat on the cut floor ---------------------------------------
FIND_SIZE = (0.090, 0.026, 0.016)  # long axis first; 16 mm tall
FIND_MASS = 0.035
FIND_X, FIND_Y = 0.265, 0.050
FIND_YAW = math.radians(20.0)  # long axis turned 20 deg from +x toward +y

# -- the trowel: a loose grey block parallel to the find, 18 mm from its -y long side, handle end (-x) toward the robot
TROWEL_SIZE = (0.140, 0.040, 0.018)
TROWEL_MASS = 0.090
TROWEL_X, TROWEL_Y = 0.2543, -0.0082
TROWEL_YAW = math.radians(20.0)

# -- the tool tray (fixed): open orange tray, inside 180 x 130 mm, 10 mm floor, 5 mm walls rising to 30 mm ------------
TRAY_X, TRAY_Y = 0.25, -0.36
TRAY_OUTER = (0.190, 0.140)
TRAY_FLOOR_H = 0.010
TRAY_WALL_T = 0.005
TRAY_WALL_H = 0.020  # on the floor: the rim is 30 mm above the table
TRAY_INNER = (TRAY_X - 0.090, TRAY_X + 0.090, TRAY_Y - 0.065, TRAY_Y + 0.065)  # x 0.16-0.34, y -0.425..-0.295
TRAY_FLOOR_Z = TOP + TRAY_FLOOR_H

# -- the photo-kit box (fixed): open dark-grey tray, inside 140 x 90 mm, 5 mm floor, 5 mm walls rising to 15 mm ------
KIT_X, KIT_Y = 0.25, 0.33
KIT_OUTER = (0.150, 0.100)
KIT_FLOOR_H = 0.005
KIT_WALL_T = 0.005
KIT_WALL_H = 0.010
KIT_FLOOR_Z = TOP + KIT_FLOOR_H

# -- the scale: a loose white bar standing in for a 10 cm photographic scale, flat in the kit, long axis along x --------
SCALE_SIZE = (0.100, 0.020, 0.015)
SCALE_MASS = 0.025
SCALE_X, SCALE_Y = KIT_X, KIT_Y
SCALE_YAW = 0.0

# -- rule thresholds (scoring.py) -------------------------------------------------------------------------------------
# Disturbed (latched over the whole trial), against the find's pose at the first control step.
FIND_MOVE = 0.005  # its centre more than this from the reference, horizontally
FIND_TURN_DEG = 3.0  # its heading turned more than this
FIND_TILT_DEG = 3.0  # tilted more than this
FIND_LIFT = 0.003  # its bottom more than this above the cut floor
# Rests on the find (at the end): the scale's or the trowel's outline overlaps the find's, bottom below this height.
REST_ON_FIND_H = FIND_SIZE[2] + 0.005  # 21 mm: the find's 16 mm top plus 5 mm
# On the floor of the cut.
FLOOR_TOL = 0.003  # the scale's bottom within this of the cut floor
FLAT_DEG = 5.0  # its 15 mm axis within this of vertical
CUT_SLACK = 0.002  # its whole outline inside the cut opening, with this much slack (contact penetration)
# The scale against the find (find frame: u along its long axis, n square to it).
HALF_WIDTHS = FIND_SIZE[1] / 2 + SCALE_SIZE[1] / 2  # 23 mm: g = |offset along n| - this
TIGHT_A_DEG, TIGHT_G, TIGHT_S = 5.0, (0.010, 0.040), 0.025
ROUGH_A_DEG, ROUGH_G, ROUGH_S = 12.0, (0.005, 0.050), 0.045
# Stowed: the trowel's centre inside the tray's inner outline, its bottom within this of the tray floor.
STOW_TOL = 0.005
# Hands clear (at the end), read with rigstate.
HALF_OPEN = 0.5  # both grippers open at least this much (0 closed .. 1 open, 93.9 mm): pads 47 mm or more apart
# A fully open gripper's pads reach at most about 56 mm from its TCP seen from above (half the 94 mm gap, the 8 mm
# pad, and half its 20 mm width): with the TCP 60 mm or more outside the frame, no pad is over the frame. The zone
# runs on past the frame's far side: a TCP out there has its arm reaching over the frame.
CLEAR_MARGIN = 0.060
CLEAR_BOX = (FRAME[0] - CLEAR_MARGIN, math.inf,
             FRAME[2] - CLEAR_MARGIN, FRAME[3] + CLEAR_MARGIN)  # no TCP at x 0.05 or more with y -0.21..+0.21
FULL_GAP = 0.0939  # finger gap fully open (m)
# Success: the 1.0 state, still: each loose object's centre within this of where it was over the last 0.5 s.
STILL_WINDOW_S = 0.5
STILL_MOVE = 0.003
# A new episode: task.py passes rigstate's episode number, and the tracker starts over only when it changes. Only the
# scorer used without it (episode None) falls back to this: all three loose objects on their starting poses (within
# START_TOL and START_TURN_DEG) and at least one of them moved more than RESET_JUMP in one control step.
START_TOL = 0.002
START_TURN_DEG = 2.0
RESET_JUMP = 0.020

# Colours (sRGB).
BANK_BROWN = (0.62, 0.48, 0.33)
FIND_CREAM = (0.93, 0.89, 0.78)
TROWEL_GREY = (0.45, 0.45, 0.47)
TRAY_ORANGE = (0.95, 0.50, 0.10)
KIT_GREY = (0.30, 0.31, 0.33)
SCALE_WHITE = (0.97, 0.97, 0.97)
