"""Every dimension of the task in one place: metres, kilograms, seconds, real size. NumPy only.

World frame: x forward from the line through the two arm bases, y to the robot's left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are ABOVE THE TABLE TOP.

Scene ("est" marks a figure not read at a source):
- a broken 12 oz ceramic mug in four rigid pieces on a counter mat in front of the robot: its base (a disc with a
  ring of wall, 8 cm across), its handle (with a strip of wall), and two curved pieces of wall, every broken edge
  sharp. The pieces are simplified as boxes and cylinders; real trials use them 3D printed from the same shapes;
- on the robot's right, an empty rigid clear storage box (outside 14 x 28 x 9 cm, est), its lid hinged on the far
  edge and standing open, a magnetic catch at the front;
- on the robot's left, the kitchen wastebasket with a white bag in it (rigid, opening 21 x 19 cm, rim 18 cm up, est).
The box and the can stand on a non-slip mat: fixed. The pieces are loose.
"""

from __future__ import annotations

import math

TOP = 0.7475
CONTROL_HZ = 20.0
EPS = 1e-9

# -- the four pieces (loose) ----------------------------------------------------------------------------------------
PIECES = ("mug_base", "mug_handle", "wall_piece_a", "wall_piece_b")
PIECE_FRICTION = (0.55, 0.45)  # glazed ceramic on a rubber mat (est)

BASE_R, BASE_FLOOR, BASE_WALL, BASE_H = 0.040, 0.006, 0.006, 0.025  # 8 cm across, wall 25 mm tall
BASE_MASS = 0.12
HANDLE_STRIP = (0.070, 0.030, 0.006)  # x, y, z: the strip of wall the handle is still on, lying flat
HANDLE_POST = (0.010, 0.012, 0.030)
HANDLE_BAR = (0.060, 0.012, 0.008)
HANDLE_MASS = 0.06
SHARD = (0.070, 0.050, 0.020)  # x, y, height: a curved piece of wall lying concave side down, as three facets
SHARD_T = 0.006
SHARD_MASS = 0.04

START = {  # (x, y, yaw) of each piece's frame origin (on the mat, centred on its footprint)
    "mug_base": (0.230, 0.045, 0.0),
    "mug_handle": (0.310, -0.022, 0.0),
    "wall_piece_a": (0.230, -0.032, 0.0),
    "wall_piece_b": (0.310, 0.042, 0.0),
}

# -- the parts box (fixed): a rigid clear storage box, lid hinged on its far edge, a snap latch at the front ----------
BOX_C = (0.300, -0.235)
BOX_OUT = (0.140, 0.280, 0.090)  # x, y, height
BOX_WALL = 0.004
BOX_FLOOR = 0.004
BOX_X0, BOX_X1 = BOX_C[0] - BOX_OUT[0] / 2, BOX_C[0] + BOX_OUT[0] / 2  # 0.230, 0.370
BOX_Y0, BOX_Y1 = BOX_C[1] - BOX_OUT[1] / 2, BOX_C[1] + BOX_OUT[1] / 2  # -0.375, -0.095
BOX_IN_X = (BOX_X0 + BOX_WALL, BOX_X1 - BOX_WALL)
BOX_IN_Y = (BOX_Y0 + BOX_WALL, BOX_Y1 - BOX_WALL)
RIM = BOX_OUT[2]
DROP_Y = {"mug_base": -0.327, "mug_handle": -0.263, "wall_piece_a": -0.215, "wall_piece_b": -0.145}

LID_NAME = "box_lid"
LID_LEN = BOX_OUT[0]
LID_T = 0.004
LID_MASS = 0.190  # 4 mm acrylic, 14 x 28 cm
LID_OPEN_DEG = 100.0  # as drawn: on its stop 10 degrees past upright
LID_DAMP = 0.002
LID_HINGE_Z = RIM + LID_T / 2
LID_SHUT_DEG = 5.0
# a magnetic catch at the middle of the front rim (cabinet hardware, about 2 kg pull, est) holds the shut lid

# -- the trash can (fixed, open) ----------------------------------------------------------------------------------
CAN_C = (0.330, 0.270)
CAN_OUT = (0.220, 0.200, 0.180)  # x, y, height: a 6 L open kitchen wastebasket (est)
CAN_WALL = 0.006
CAN_IN_X = (CAN_C[0] - CAN_OUT[0] / 2 + CAN_WALL, CAN_C[0] + CAN_OUT[0] / 2 - CAN_WALL)
CAN_IN_Y = (CAN_C[1] - CAN_OUT[1] / 2 + CAN_WALL, CAN_C[1] + CAN_OUT[1] / 2 - CAN_WALL)
CAN_RIM = CAN_OUT[2]

# every hinged panel the closed pads push shut: name -> (hinge x, inward sign along x, hinge z, length, y range, open)
PANELS = {
    LID_NAME: (BOX_X1, -1.0, LID_HINGE_Z, LID_LEN, (BOX_Y0, BOX_Y1), LID_OPEN_DEG),
}
BALANCE_DEG = 90.0

# -- scoring thresholds -----------------------------------------------------------------------------------------------
IN_ALLOW = 0.002
RELEASE_CLEAR = 0.040
OPEN_THRESHOLD = 0.85
STILL_WINDOW_S = 1.0
WIN_STEPS = int(round(STILL_WINDOW_S * CONTROL_HZ))
STILL_MOVE = 0.003
OFF_TABLE_DROP = 0.010
TABLE_X = (0.36 - 0.55, 0.36 + 0.55)
TABLE_Y = (-0.65, 0.65)

# -- colours (sRGB) ---------------------------------------------------------------------------------------------------
GLAZE = (0.90, 0.88, 0.82)  # cream glaze
BISQUE = (0.78, 0.66, 0.55)  # the broken edges
BOX_CLEAR = (0.82, 0.88, 0.92)
LATCH_BLUE = (0.20, 0.42, 0.75)
CAN_GREY = (0.55, 0.57, 0.60)
BAG_WHITE = (0.96, 0.96, 0.96)
MAT_GREY = (0.25, 0.26, 0.28)


def flap_dir(name: str, theta_deg: float) -> tuple[float, float]:
    t = math.radians(theta_deg)
    return PANELS[name][1] * math.cos(t), math.sin(t)


def flap_hinge_x(name: str) -> float:
    return PANELS[name][0]
