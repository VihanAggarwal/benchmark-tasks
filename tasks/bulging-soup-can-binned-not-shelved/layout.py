"""Every dimension of the task in one place: metres, kilograms, seconds, real size. NumPy only.

World frame: x forward from the line through the two arm bases, y to the robot's left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are ABOVE THE TABLE TOP.

Scene ("est" marks a figure not read at a source):
- three 8 oz cans of tomato sauce (Hunt's Tomato Sauce 8 oz: a 211 x 304 can, 2 11/16 in across by 3 4/16 in
  tall, 68 x 83 mm) standing in a row on the counter; the right-hand one (nearest the bin) has its top
  lid domed 5 mm (est), a bulging can;
- on the robot's left a pantry shelf riser (fixed);
- on the robot's right a small lidded countertop trash bin (fixed), its lid hinged along the far edge, standing open
  on its stop 10 degrees past upright; pushed past upright it falls shut under its own weight.
"""

from __future__ import annotations

import math

TOP = 0.7475
CONTROL_HZ = 20.0
EPS = 1e-9

# -- the cans (loose) ---------------------------------------------------------------------------------------------------
CAN_R = 0.034
CAN_H = 0.083
CAN_MASS = 0.27  # 227 g of sauce plus the steel can (est)
CAN_FRICTION = (0.5, 0.4)
DOME_H = 0.005  # the bulging can's domed top lid (est)
CANS = ("bulged_can", "can_a", "can_b")
CAN_XY0 = {"can_a": (0.280, 0.100), "can_b": (0.280, 0.020), "bulged_can": (0.280, -0.060)}  # a row along y,
#   8 cm apart; the bulging can is the right-hand one
GOOD = ("can_a", "can_b")
BULGED = "bulged_can"

# -- the pantry shelf riser (fixed) ---------------------------------------------------------------------------------------
SHELF_X = (0.300, 0.460)
SHELF_Y = (0.150, 0.400)
SHELF_TOP = 0.040
SHELF_SPOTS = {"can_a": (0.380, 0.205), "can_b": (0.380, 0.325)}  # where the demo stands the good cans

# -- the lidded trash bin (fixed) -----------------------------------------------------------------------------------------
BIN_C = (0.270, -0.260)
BIN_OUT = (0.180, 0.200, 0.100)  # x, y, height (rim 10 cm up)
BIN_WALL = 0.005
BIN_FLOOR = 0.005
BIN_X0, BIN_X1 = BIN_C[0] - BIN_OUT[0] / 2, BIN_C[0] + BIN_OUT[0] / 2  # 0.180, 0.360
BIN_Y0, BIN_Y1 = BIN_C[1] - BIN_OUT[1] / 2, BIN_C[1] + BIN_OUT[1] / 2  # -0.360, -0.160
BIN_IN_X = (BIN_X0 + BIN_WALL, BIN_X1 - BIN_WALL)
BIN_IN_Y = (BIN_Y0 + BIN_WALL, BIN_Y1 - BIN_WALL)
BIN_RIM = BIN_OUT[2]

LID_NAME = "bin_lid"
LID_LEN = BIN_OUT[0]
LID_T = 0.006
LID_MASS = 0.12
LID_OPEN_DEG = 100.0  # as drawn: 10 degrees past upright, leaning back, on its stop
LID_HINGE_Z = BIN_RIM + LID_T / 2
LID_DAMP = 0.002
LID_SHUT_DEG = 5.0
BALANCE_DEG = 90.0

# every hinged panel the closed pads push shut: name -> (hinge x, inward sign along x, hinge z, length, y range, open)
PANELS = {LID_NAME: (BIN_X1, -1.0, LID_HINGE_Z, LID_LEN, (BIN_Y0, BIN_Y1), LID_OPEN_DEG)}

# -- scoring thresholds ---------------------------------------------------------------------------------------------------
IN_ALLOW = 0.002
UPRIGHT_DEG = 15.0
ON_SHELF_GAP = 0.010  # a shelved can's bottom within 10 mm of the shelf top
RELEASE_CLEAR = 0.040
OPEN_THRESHOLD = 0.85
STILL_WINDOW_S = 1.0
WIN_STEPS = int(round(STILL_WINDOW_S * CONTROL_HZ))
STILL_MOVE = 0.003
OFF_TABLE_DROP = 0.010
TABLE_X = (0.36 - 0.55, 0.36 + 0.55)
TABLE_Y = (-0.65, 0.65)

# -- colours (sRGB) -------------------------------------------------------------------------------------------------------
TIN = (0.78, 0.79, 0.80)
RIM_STEEL = (0.52, 0.53, 0.55)  # the darker seamed rim band round every can's top edge
LABEL_RED = (0.78, 0.12, 0.10)
LABEL_WHITE = (0.94, 0.93, 0.90)
WOOD = (0.66, 0.50, 0.33)
BIN_WHITE = (0.93, 0.93, 0.92)
LID_GREY = (0.55, 0.57, 0.60)


def flap_dir(name: str, theta_deg: float) -> tuple[float, float]:
    t = math.radians(theta_deg)
    return PANELS[name][1] * math.cos(t), math.sin(t)
