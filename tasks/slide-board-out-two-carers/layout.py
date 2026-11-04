"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The bed deck: a fixed white platform standing in for the mattress surface after a lateral transfer.
DECK_SIZE = (0.200, 0.440, 0.030)  # along x, along y, height
DECK_X, DECK_Y = 0.28, -0.04  # its footprint spans x 0.18..0.38, y -0.26..+0.18
DECK_TOP = TOP + DECK_SIZE[2]  # z of the deck's top face (the "bed")
DECK_HALF_X, DECK_HALF_Y = DECK_SIZE[0] / 2, DECK_SIZE[1] / 2

# The slide (transfer) board: a loose, hollow, light-blue polymer board lying flat on the deck, long side along y.
BOARD_SIZE = (0.040, 0.320, 0.020)  # along x, along y, thickness
BOARD_MASS = 0.10  # a hollow polymer board at this size (0.39 g/cm3 overall)
BOARD_X, BOARD_Y = 0.28, 0.02  # it spans y -0.14..+0.18: its left end at the deck's left edge
BOARD_TOP = DECK_TOP + BOARD_SIZE[2]

# The patient: a loose skin-pink block standing in for the patient's torso, lying on the board near its left end.
PATIENT_SIZE = (0.070, 0.060, 0.040)  # along x, along y, height; it overhangs the 40 mm board by 15 mm each side in x
PATIENT_MASS = 0.20
PATIENT_X, PATIENT_Y = 0.28, 0.10  # it spans y +0.07..+0.13; the board runs 210 mm beyond it toward -y
PATIENT_BOTTOM_Z = BOARD_TOP  # its underside rests on the board, 20 mm above the deck

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper (13 mm past the TCP to 25 mm
#                     behind it; the flat pad faces are about 22 mm of that)

# Rule thresholds (scoring.py). The patient rules hold at every control step of the trial.
MOVED_MAX = 0.015  # the patient's centre this far from its start, horizontally, at any moment: 0 (dragged, pushed,
#                    carried aside)
RAISE_MAX = 0.030  # the centre of the patient's underside this far above where it started (the board's top), at any
#                    moment: 0 (lifted away instead of steadied)
TILT_MAX_DEG = 10.0  # the patient tilted more than this at any moment: 0 (tipped over the board's edge)
DESCENT_MAX = 0.15  # the patient's downward speed above this at any moment: 0 (dropped; a fall from the board's
#                     20 mm reaches 0.63 m/s, from 2 mm 0.20 m/s; the demo lowers it at 0.012 m/s)
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table: 0
# The board's progress, from the two outlines seen from above (world-aligned bounding boxes).
OUT_CLEAR = 0.010  # 'fully out': the board's outline at least this far from the patient's along y (no overlap)
HALF_OUT_SHARE = 0.5  # 'at least half out' (0.5): no more than this share of the board's starting footprint under
#                       the patient (40 x 60 mm) is still under it (a straight pull along y of 80 mm or more, since
#                       the board starts 50 mm past the patient; across, along x, 35 mm or more, since the patient
#                       overhangs the board by 15 mm each side)
PULL_MIN = 0.020  # 'started' (0.25): the board's centre this far along -y from where it lay
# 'Resting on the deck' (the board at the end) and 'flat on the deck' (the patient at the end).
ON_DECK_Z_TOL = 0.003  # lowest point within this of the deck's top
BOARD_FLAT_DEG = 3.0  # the board tilted less than this (either broad face up)
PATIENT_FLAT_DEG = 5.0  # the patient tilted less than this
# 'Let go' (rigstate): a gripper this open (fingers 84.5 mm apart) holds nothing, or, if less open, its finger zone
# is at least LET_GO_GAP from the item it handles (left: the patient; right: the board).
GRIPPER_OPEN = 0.9
LET_GO_GAP = 0.010
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
WHITE = (0.95, 0.95, 0.95)
LIGHT_BLUE = (0.55, 0.75, 0.95)
SKIN_PINK = (0.93, 0.72, 0.65)
