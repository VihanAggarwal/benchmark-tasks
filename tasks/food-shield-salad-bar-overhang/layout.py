"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases (toward the customers' side of the bar), y to the
left, z up. The arm bases stand at x 0, y +-0.305; the station's table top (the salad bar's counter) is at z 0.7475.
The bar is drawn at 1:4: 45 to 75 mm here is 18 to 30 cm on a real bar.
"""

import math

TOP = 0.7475  # station table top: the counter

# Five 1/3-size food pans in a row along y (a countertop cold-pan unit, fixed): each a stainless pan 81 mm deep (x)
# and 44 mm wide (y), 22 mm tall, with its food filling it to a 3 mm layer on top inset 2 mm from its rim, so the
# food's top is 25 mm above the counter. 2 mm between pans (the adapter bars). The pans' customer-side edge is the
# line the overhang is measured from.
PAN_X0, PAN_X1 = 0.259, 0.340  # staff-side edge, customer-side edge
PAN_X = (PAN_X0 + PAN_X1) / 2
PAN_DEPTH = PAN_X1 - PAN_X0  # 0.081
PAN_W = 0.044
PAN_PITCH = 0.046
PAN_BODY_H = 0.022
FOOD_INSET = 0.002
FOOD_H = 0.003
PAN_TOP = PAN_BODY_H + FOOD_H  # 0.025 above the counter
PANS = (  # (name, centre y, food colour key), left (+y) to right
    ("lettuce", 2 * PAN_PITCH, "LETTUCE"),
    ("tomato", PAN_PITCH, "TOMATO"),
    ("cucumber", 0.0, "CUCUMBER"),
    ("sweetcorn", -PAN_PITCH, "SWEETCORN"),
    ("croutons", -2 * PAN_PITCH, "CROUTONS"),
)
ROW_HALF = 2 * PAN_PITCH + PAN_W / 2  # 0.114: the row runs y -0.114 .. +0.114
ROW_ENDS = (ROW_HALF, -ROW_HALF)  # where the overhang is read (a tape at each end of the row)

# Tray slide: a fixed stainless rail on the customers' side, 94 mm past the pans' edge. It stops a shield pushed all
# the way forward at an overhang of 94 mm (outside every band), so the 75-90 mm band is physically reachable.
SLIDE_X0 = PAN_X1 + 0.094  # 0.434
SLIDE_SIZE = (0.010, 0.320, 0.010)
SLIDE_X = SLIDE_X0 + SLIDE_SIZE[0] / 2

# The portable food shield: one loose rigid body. Its base frame is at the centre of its footprint, on the end panels'
# undersides (z 0 local), local +x toward its front lip (the customers' side). Clear acrylic top glass on two acrylic
# end panels, an acrylic front lip under the glass's front edge, a stainless end cap on the glass over each panel.
SHIELD_D = 0.080  # front to back (local x)
SHIELD_W = 0.320  # end to end (local y): the panels' outer faces at +-0.160
GLASS_SIZE = (SHIELD_D, SHIELD_W, 0.004)
GLASS_Z0 = 0.090  # its underside; top 0.094
PANEL_SIZE = (SHIELD_D, 0.004, GLASS_Z0)  # 0-90 mm up
PANEL_Y = SHIELD_W / 2 - PANEL_SIZE[1] / 2  # 0.158
PANEL_INNER = PANEL_Y - PANEL_SIZE[1] / 2  # 0.156: the panels' inner faces
LIP_SIZE = (0.004, 2 * PANEL_INNER, 0.025)  # between the panels, 65-90 mm up, flush with the glass's front edge
LIP_Z0 = GLASS_Z0 - LIP_SIZE[2]  # 0.065
LIP_X = SHIELD_D / 2 - LIP_SIZE[0] / 2  # 0.038
LIP_FACE_X = SHIELD_D / 2  # 0.040: the lip's outer face (what the tape reads to)
LIP_HALF = PANEL_INNER  # the lip runs y -0.156 .. +0.156 local
LIP_MID_Z = LIP_Z0 + LIP_SIZE[2] / 2  # 0.0775
CAP_SIZE = (0.030, 0.024, 0.025)  # on the glass, 94-119 mm up, centred over each panel's mid-depth
CAP_Y = 0.148
CAP_Z0 = GLASS_Z0 + GLASS_SIZE[2]  # 0.094
SHIELD_H = CAP_Z0 + CAP_SIZE[2]  # 0.119
# 0.25 kg: the acrylic parts weigh 228 g at 1.19 g/cm3, and each end cap is a hollow cover of 0.5 mm stainless
# sheet (about 13 g), not a solid block (a solid steel cap would be 142 g).
SHIELD_MASS = 0.25
# Local shapes (centre, size, colour key) of the shield, at rest.
SHIELD_BOXES = (
    ((0.0, 0.0, GLASS_Z0 + GLASS_SIZE[2] / 2), GLASS_SIZE, "ACRYLIC"),
    ((0.0, PANEL_Y, PANEL_SIZE[2] / 2), PANEL_SIZE, "ACRYLIC"),
    ((0.0, -PANEL_Y, PANEL_SIZE[2] / 2), PANEL_SIZE, "ACRYLIC"),
    ((LIP_X, 0.0, LIP_Z0 + LIP_SIZE[2] / 2), LIP_SIZE, "LIP"),
    ((0.0, CAP_Y, CAP_Z0 + CAP_SIZE[2] / 2), CAP_SIZE, "STEEL"),
    ((0.0, -CAP_Y, CAP_Z0 + CAP_SIZE[2] / 2), CAP_SIZE, "STEEL"),
)
SHIELD_BBOX_CENTRE = (0.0, 0.0, SHIELD_H / 2)  # local centre of the box around every shape (what offline carries)

# Start: pulled back to the staff side for refilling, square to the bar, lip toward the customers, and left 30 mm
# toward the left end of the bar. Carried straight forward from here, it would end 30 mm off centre.
START = (0.170, 0.030)
START_YAW = 0.0

# Rule thresholds (scoring.py).
STAND_TOL = 0.003  # 'standing on the counter': every corner of both panels' undersides within this of the counter
TILT_MAX_DEG = 5.0
TURN_MAX_DEG = 10.0  # the lip's direction within this of straight toward the customers (+x)
OVERHANG_FULL = (0.045, 0.075)  # both readings in this band: 0.75 (18-30 cm at 1:4)
OVERHANG_PART = (0.025, 0.090)  # both in this band (and not both in the full one): 0.25
OVERHANG_AIM = 0.060
CENTRE_TOL = 0.015  # the shield's centre within this of the row's centre along the bar (y 0): +0.25
REST_WINDOW_S = 0.5  # 'at rest': neither panel's foot moved more than REST_MOVE in the last 0.5 s
REST_MOVE = 0.003
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the shield moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)
EPS = 1e-9

# Colours (sRGB).
STEEL = (0.78, 0.79, 0.81)
PAN_STEEL = (0.70, 0.71, 0.73)
ACRYLIC = (0.72, 0.82, 0.88)  # clear acrylic, drawn light blue-grey (opaque in the simulator)
LIP = (0.60, 0.72, 0.80)
LETTUCE = (0.30, 0.62, 0.20)
TOMATO = (0.80, 0.16, 0.12)
CUCUMBER = (0.70, 0.85, 0.55)
SWEETCORN = (0.95, 0.80, 0.20)
CROUTONS = (0.78, 0.60, 0.36)


def colour(key: str):
    return globals()[key]


def overhang_band_deg_limit() -> float:
    """How far the shield can be turned with both readings still inside the 30 mm full band: atan(30 / 228)."""
    return math.degrees(math.atan((OVERHANG_FULL[1] - OVERHANG_FULL[0]) / (2 * ROW_HALF)))
