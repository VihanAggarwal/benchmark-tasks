"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A rigid two-piece setup box (greyboard, 2 mm, paper wrapped): an open base 70 x 114 x 40 mm and a telescoping lid
(top panel 2 mm, skirt 34 mm below it) whose inside is 2.5 mm larger than the base's outside on every side. A
gift (an oak block) lies in the base. The base stands where it is packed, the lid stands
top up on the table beside it. Seated: the lid's top panel rests on the base's rim, the skirt over the base's
walls by 34 mm.

Figures follow the cut list in task.json's setup and the README.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the base (loose) ----------------------------------------------------------------------------------------------
# Two-piece rigid ("setup") box: 2 mm greyboard wrapped in paper. Sizes follow the README cut list; check the
# built box with calipers.
BASE_LX, BASE_LY, BASE_H = 0.070, 0.114, 0.040  # outside: x across, y along, z tall
WALL = 0.002
FLOOR = 0.002
BASE_X, BASE_Y = 0.330, -0.070  # centre of the base's footprint
BASE_MASS = 0.050  # kg: about 44 cm3 of greyboard at 1.1 g/cm3 (est)
BASE_COLOR = (0.09, 0.09, 0.10)
INNER_LX, INNER_LY = BASE_LX - 2 * WALL, BASE_LY - 2 * WALL  # 66 x 110 mm

# -- the lid (loose) -----------------------------------------------------------------------------------------------
CLR = 0.0025  # play between the lid's inside and the base's outside, each side (design play; checked by feeler gauge on each built box)
LID_IX, LID_IY = BASE_LX + 2 * CLR, BASE_LY + 2 * CLR  # inside 75 x 119 mm
LID_OX, LID_OY = LID_IX + 2 * WALL, LID_IY + 2 * WALL  # outside 79 x 123 mm
LID_PANEL = 0.002
LID_SKIRT = 0.034  # the skirt below the panel's inner face
LID_H = LID_PANEL + LID_SKIRT  # 36 mm tall, standing top up
LID_X, LID_Y = 0.330, -0.290  # where it stands at the start (centre of its footprint)
LID_MASS = 0.050  # kg (est)
LID_COLOR = (0.12, 0.12, 0.13)
# its base frame sits at the centre of the skirt's lower rim, z up through the panel
SEAT_TOP = BASE_H + LID_PANEL  # 42 mm: the lid's top outer face when seated
SEAT_RIM = BASE_H - LID_SKIRT  # 6 mm: the skirt's rim above the table when seated

# -- the product (loose) -------------------------------------------------------------------------------------------
# The gift: an oak block, lying flat in the base, long side along y (README).
PR_LX, PR_LY, PR_H = 0.055, 0.105, 0.028
PR_MASS = 0.120
PR_COLOR = (0.93, 0.74, 0.08)
PR_Z = FLOOR + PR_H / 2  # centre height over the table

# -- friction (est) ------------------------------------------------------------------------------------------------
FRICTION = (0.45, 0.40)  # board on board and on the table (simulator setting)

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------------
SEAT_TOL = 0.0015  # 1.0: the lid's top outer face within this of 42 mm
SQUARE_YAW_DEG = 1.5  # 1.0: the lid's yaw within this of the base's
SQUARE_TILT_DEG = 1.0  # 1.0: the lid's top face tilted under this
CENTRE_TOL = 0.0035  # 1.0: the lid's centre within this of the base's (top view); 1.0 mm more than the play
ENGAGED_OVERLAP = 0.010  # 0.5: the skirt's lowest point this far below the base's rim
ENGAGED_CENTRE = 0.008
ENGAGED_TILT_DEG = 10.0
ON_BASE_CENTRE = 0.025  # 0.25: the lid resting on the base, its centre within this of the base's
ON_BASE_TILT_DEG = 30.0
ON_BASE_LOW = 0.004  # the lid's lowest point at least this far over the table (carried by the base, not standing)
BASE_MOVE = 0.020  # the base within this of where it began (top view)
BASE_LIFT = 0.003  # the base's footing within this of the table
BASE_TILT_DEG = 2.0
PRODUCT_MARGIN = 0.002  # the product's centre inside the base's inside, with this much slack
FALL_Z = -0.010  # latched: the lid or the base lower than this below the table top
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001
STILL_TURN_DEG = 0.3
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9

PARTS = ("base", "lid", "product")


def lid_yaw_ok_limit_deg() -> float:
    """The largest yaw (deg) at which the lid's inside still clears the base's outside corners (geometric limit)."""
    best = 0.0
    for i in range(1, 4000):
        t = math.radians(i * 0.001)
        need_x = BASE_LX * math.cos(t) + BASE_LY * math.sin(t)
        need_y = BASE_LY * math.cos(t) + BASE_LX * math.sin(t)
        if need_x <= LID_IX and need_y <= LID_IY:
            best = i * 0.001
    return best
