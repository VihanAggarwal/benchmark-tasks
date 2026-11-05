"""Every dimension of the task, in one place (metres, kilograms, radians, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The manikin is one loose mechkit assembly: its base body is the trunk, and its one link, the head, turns on a pin
hinge (the neck) about the trunk's x axis. Its shapes are given in the manikin's base frame with the neck straight;
that frame's origin is the middle of the trunk's bottom face.
"""

import math

TOP = 0.7475  # station table top

# The grey floor mat the casualty lies on: a thin fixed rubber sheet.
MAT_X, MAT_Y = 0.200, -0.035
MAT_SIZE = (0.120, 0.380, 0.003)
MAT_TOP = 0.003  # above the table top
MAT_RECT = (MAT_X - MAT_SIZE[0] / 2, MAT_X + MAT_SIZE[0] / 2, MAT_Y - MAT_SIZE[1] / 2, MAT_Y + MAT_SIZE[1] / 2)

# The manikin (base frame: origin at the middle of the trunk's bottom face, x across, y toward the head, z up).
MANIKIN_ORIGIN = (0.200, -0.085, TOP + MAT_TOP)
TRUNK_CENTER = (0.0, 0.0, 0.020)
TRUNK_SIZE = (0.070, 0.240, 0.040)
TRUNK_MASS = 0.22
NECK_ANCHOR = (0.0, 0.120, 0.020)  # the hinge: along the trunk's x axis, at the head end of the trunk, mid-height
NECK_BAR_CENTER = (0.0, 0.135, 0.020)  # the neck: a short bar between trunk and head (part of the head link)
NECK_BAR_SIZE = (0.024, 0.030, 0.020)
HEAD_CENTER = (0.0, 0.185, 0.022)
HEAD_SIZE = (0.060, 0.070, 0.044)
HEAD_MASS = 0.07  # the head link: head and neck bar together
NECK_LIMIT = 0.70  # rad (40 deg) each way
NECK_SPRING = 0.02  # N m/rad toward straight
NECK_DAMPING = 0.005  # N m s/rad
MASS_TOTAL = TRUNK_MASS + HEAD_MASS

# The stretcher: one fixed slab 30 mm tall, built from three boxes of the same height so its top is flush; the
# white head zone is the middle one.
STRETCHER_X = 0.365
STRETCHER_WIDTH = 0.110  # along x
STRETCHER_Y0, STRETCHER_Y1 = -0.235, 0.165
STRETCHER_TOP = 0.030  # above the table top
HEAD_ZONE_Y0, HEAD_ZONE_Y1 = 0.065, 0.135
STRETCHER_RECT = (STRETCHER_X - STRETCHER_WIDTH / 2, STRETCHER_X + STRETCHER_WIDTH / 2, STRETCHER_Y0, STRETCHER_Y1)
HEAD_TARGET = (STRETCHER_X, (HEAD_ZONE_Y0 + HEAD_ZONE_Y1) / 2)  # (0.365, 0.100): the head zone's centre

# Rule thresholds (scoring.py).
LIFTED = 0.005  # a body is lifted when its lowest point is more than this above the surface under its middle
LIFT_ALONE = 0.015  # 0: one body lifted more than this while the other is not lifted (not moved as one unit)
LIFT_TOGETHER = 0.020  # 0.25 off the stretcher: both bodies lifted this much at once, neck within NECK_TOGETHER_DEG
LIFT_CAP = 0.080  # 0: a body's lowest point more than this above the table top
OFF_TABLE = 0.010  # 0: a body's lowest point more than this below the table top
FALL_SPEED = 0.25  # m/s: 0 if a body's lowest point drops faster than this between two control steps
NECK_ZERO_DEG = 20.0  # 0: the neck ever bent more than this
NECK_ONE_DEG = 6.0  # 1.0: the neck never bent more than this ...
FINAL_ONE_DEG = 3.0  # ... and within this at the end
NECK_075_DEG = 10.0
FINAL_075_DEG = 5.0
NECK_05_DEG = 15.0
NECK_TOGETHER_DEG = 10.0
# 'Held by the neck': a gripper's tool point within NECK_ZONE_ALONG of the neck bar's centre along the manikin's
# length, NECK_ZONE_ACROSS across it and no more than NECK_ZONE_ABOVE above it, while either body is lifted.
NECK_ZONE_ALONG = 0.025
NECK_ZONE_ACROSS = 0.050
NECK_ZONE_ABOVE = 0.060
# 'On the stretcher'.
ON_LOW_BELOW, ON_LOW_ABOVE = 0.003, 0.004  # each body's lowest point this close to the stretcher's top
HEAD_TOL = 0.025  # the head box's centre from the head zone's centre, horizontally
FLAT_DEG = 10.0  # the trunk lying flat on its back (its back-to-front axis within this of straight up)
# 'Let go' for 1.0: each gripper open and its finger-tip point this far from the manikin.
GRIP_OPEN = 0.9
CLEAR_DIST = 0.030
# Stillness for success (SUCCESS_HOLD_S in task.py): trunk speed, trunk spin, neck speed, head speed.
STILL_SPEED = 0.01
STILL_SPIN = 0.5
STILL_NECK = 0.1
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the manikin moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
MAT_GREY = (0.40, 0.40, 0.42)
SKIN = (0.85, 0.70, 0.58)
NECK_BEIGE = (0.68, 0.53, 0.42)
STRETCHER_BLUE = (0.15, 0.30, 0.65)
HEAD_ZONE_WHITE = (0.92, 0.92, 0.92)

DEG = math.pi / 180.0
