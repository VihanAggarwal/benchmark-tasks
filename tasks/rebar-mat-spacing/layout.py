"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top

# The slab forms: four fixed plywood boards closing a rectangle on the table, x 0.10 to 0.42 and y -0.23 to
# +0.23 inside. The side forms run along x with their inner faces at y +-0.23; the front and back forms run
# along y, their inner faces at x 0.10 and x 0.42, and butt against the side forms' ends.
FORM_T = 0.010  # board thickness
FORM_H = 0.025  # board height
FORM_TOP = TOP + FORM_H
SLAB_X0, SLAB_X1 = 0.10, 0.42  # inside faces of the front and back forms
SLAB_Y = 0.23  # inside faces of the side forms, at y +-SLAB_Y
FORMS = (  # (name, centre x, centre y, size x, size y)
    ("form_left", (SLAB_X0 + SLAB_X1) / 2, SLAB_Y + FORM_T / 2, SLAB_X1 - SLAB_X0, FORM_T),
    ("form_right", (SLAB_X0 + SLAB_X1) / 2, -(SLAB_Y + FORM_T / 2), SLAB_X1 - SLAB_X0, FORM_T),
    ("form_front", SLAB_X0 - FORM_T / 2, 0.0, FORM_T, 2 * (SLAB_Y + FORM_T)),
    ("form_back", SLAB_X1 + FORM_T / 2, 0.0, FORM_T, 2 * (SLAB_Y + FORM_T)),
)

# The two bolsters: fixed grey plastic bar supports inside the forms, running along x, the bars' seats.
BOLSTER_W = 0.012  # along y
BOLSTER_H = 0.020
BOLSTER_X0, BOLSTER_X1 = 0.12, 0.40
BOLSTER_X = (BOLSTER_X0 + BOLSTER_X1) / 2
BOLSTER_LEN = BOLSTER_X1 - BOLSTER_X0
BOLSTER_YS = (0.12, -0.12)  # bolster_left, bolster_right
BOLSTER_TOP = TOP + BOLSTER_H

# The three bars: loose stand-ins for 16 mm reinforcing bars (square hollow aluminium tube painted rust brown,
# 16 x 16 mm, 400 mm long, 0.10 kg with its end caps; a solid steel bar this size would weigh 0.80 kg). Their
# size is given as they lie in the finished mat (along y); at the start each is turned a quarter turn
# (BAR_START_YAW), so it lies along x on its rack.
BAR_SIZE = (0.016, 0.400, 0.016)  # across, along, height (in the mat: along x, along y, up)
BAR_MASS = 0.10
BAR_HALF_LEN = BAR_SIZE[1] / 2
BAR_START_YAW = math.pi / 2  # lying along x
BAR_START_X = 0.28  # each bar spans x 0.08 to 0.48 on its rack
BARS = (  # (name, y) at the start: two on the left arm's side, one on the right arm's, 60 mm or more apart
    ("bar_left_inner", 0.30),
    ("bar_left_outer", 0.38),
    ("bar_right", -0.32),
)

# The racks: fixed weathered-timber blocks (dunnage) outside the forms that keep the bars off the table, two
# under each side's bars, 26 cm apart along x (each bar overhangs them by 7 cm at either end). 20 mm tall, so the
# fingers reach past a bar's underside when it is gripped at mid-height.
RACK_SIZE_X, RACK_H = 0.020, 0.020
RACK_XS = (BAR_START_X - 0.13, BAR_START_X + 0.13)  # x 0.15 and 0.41
RACKS = (  # (name, centre x, centre y, length along y)
    ("rack_left_near", RACK_XS[0], 0.34, 0.140),
    ("rack_left_far", RACK_XS[1], 0.34, 0.140),
    ("rack_right_near", RACK_XS[0], -0.32, 0.080),
    ("rack_right_far", RACK_XS[1], -0.32, 0.080),
)
RACK_TOP = TOP + RACK_H

# The intended mat (the demo lays this; the scorer accepts any three bars at the right spacing anywhere on the
# bolsters): bars along y centred on y 0 at x 0.16, 0.26 and 0.36, each 30 mm from both side forms.
MAT_XS = (0.16, 0.26, 0.36)
SPACING = 0.100  # centre to centre

# Rule thresholds (scoring.py).
REST_Z_TOL = 0.003  # 'rests on' a bolster: the bar's underside within this of the bolster's top where they cross,
REST_COVER = 0.5  # and the crossing at least this share of (bar width x bolster width) seen from above
FLAT_DEG = 5.0  # 'flat': tilted less than this
HEADING_DEG = 3.0  # 'along y': long side within this of y, either way round
EDGE_DEG = 30.0  # tilted more than this: standing on an end or an edge (0)
OFF_TABLE_DROP = 0.010  # a bar's lowest point this far below the table top: it fell off the table (0)
FORM_CLEAR = 0.020  # 'across a form': its outline over a form board and its lowest point at most this above it (0)
GAP_TIGHT, GAP_LOOSE = 0.005, 0.012  # spacing within 100 mm +- this (1.0, 0.5 / 0.75)
COVER_FULL, COVER_PART = 0.020, 0.010  # end cover at least this (1.0, 0.5, 0.25 / 0.75)
STILL_SPEED, STILL_SPIN = 0.01, 0.5  # 'still': under 1 cm/s and 0.5 rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a bar that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
PLYWOOD = (0.70, 0.55, 0.35)
BOLSTER_GREY = (0.55, 0.55, 0.58)
RUST = (0.45, 0.30, 0.22)
DUNNAGE = (0.40, 0.33, 0.25)
