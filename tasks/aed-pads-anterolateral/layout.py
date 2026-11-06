"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The training manikin lies on its back with its head away from the arms, so the manikin's right side is the
arms' left (+y) and its left side is the arms' right (-y).
"""

import math

TOP = 0.7475  # station table top

# The torso: a fixed, armless, flat-topped training torso (tan shell), its chest top 60 mm above the table.
TORSO_X0, TORSO_X1 = 0.150, 0.470  # x 0.470 is the shoulder line (the torso's edge toward the head)
TORSO_HALF_Y = 0.190
TORSO_H = 0.060
TORSO_SIZE = (TORSO_X1 - TORSO_X0, 2 * TORSO_HALF_Y, TORSO_H)  # 320 x 380 x 60 mm
TORSO_X = (TORSO_X0 + TORSO_X1) / 2
CHEST_TOP = TOP + TORSO_H

# Neck and head (fixed, out of reach; they show which way the manikin lies).
NECK_SIZE = (0.020, 0.100, 0.050)
NECK_X = TORSO_X1 + NECK_SIZE[0] / 2  # x 0.470 .. 0.490
HEAD_RADIUS, HEAD_H = 0.075, 0.090
HEAD_X = NECK_X + NECK_SIZE[0] / 2 + HEAD_RADIUS + 0.001  # 1 mm clear of the neck

# Landmarks drawn on the chest: thin (1 mm) dark-brown markings, fixed.
MARK_T = 0.001
NOTCH_SIZE = (0.020, 0.020, MARK_T)  # the sternal notch, centred on the shoulder line's middle
NOTCH_X = 0.455
STERNUM_SIZE = (0.120, 0.010, MARK_T)  # the breastbone line, along the midline below the notch
STERNUM_X = 0.385  # x 0.325 .. 0.445
CLAVICLE_SIZE = (0.008, 0.120, MARK_T)  # each collarbone, a line along the shoulder line beside the notch
CLAVICLE_X, CLAVICLE_Y = 0.455, 0.075  # y 0.015 .. 0.135 on each side
NIPPLE_RADIUS = 0.0075
NIPPLE_X, NIPPLE_Y = 0.340, 0.085  # (0.340, +-0.085)

# The AED unit (decor): a fixed yellow-green box between the arm bases, lid open.
AED_SIZE = (0.100, 0.140, 0.060)
AED_X, AED_Y = 0.080, 0.0

# The two training pads: loose, identical, white, lying flat on the table beside the torso.
PAD_SIZE = (0.110, 0.070, 0.018)  # long side first; 18 mm = the pad on its 15 mm closed-cell foam backing
PAD_HALF = tuple(s / 2 for s in PAD_SIZE)
PAD_MASS = 0.040
PAD_START = {  # name: (x, y, yaw); the left one lies across (long side along y), the right one along x
    "pad_left": (0.200, 0.270, math.pi / 2),
    "pad_right": (0.200, -0.270, 0.0),
}
PADS = tuple(PAD_START)

# Targets (pad centres, seen from above), from the guidelines' anterolateral placement on this torso.
# Sternal: the manikin's upper right chest, 70 mm below the shoulder line, 80 mm right of the breastbone line,
# above the nipple line. Apical: its left side below the armpit, 170 mm below the shoulder line, 145 mm left of the
# midline (45 mm in from the chest's left edge), 40 mm below and 60 mm outside the left nipple.
TARGETS = {
    "sternal": (0.400, 0.080),
    "apical": (0.300, -0.145),
}

# Rule thresholds (scoring.py).
NEAR = 0.025  # a pad's centre within this of a target: 0.5 for that target
FAR = 0.040  # ... within this: 0.25
TILT_MAX_DEG = 10.0  # 'lying on the chest': tilted at most this
ON_CHEST_TOL = 0.005  # ... and its lowest point within this of the chest top (above or below)
PAD_GAP_MIN = 0.010  # the two pads' outlines seen from above closer than this (touching, overlapping, stacked): 0
RELEASE_CLEAR = 0.010  # a pad counts only when both grippers' finger zones are at least this far from it
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a pad moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the finger pads are about 27 mm wide (across the closing axis)
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Colours (sRGB).
TAN = (0.86, 0.69, 0.55)
BROWN = (0.36, 0.22, 0.14)
PAD_WHITE = (0.95, 0.95, 0.93)
AED_GREEN = (0.55, 0.75, 0.20)
