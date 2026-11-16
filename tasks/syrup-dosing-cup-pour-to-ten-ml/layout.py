"""Every dimension of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

Parts. (1) Syrup bottle: a 4 fl oz (118 mL) HDPE children's liquid-medicine bottle, cap off, 118 mL of syrup in it.
Outside: a round body 49 mm across and 82 mm tall, a neck 28 mm across up to 105 mm; inside 2 mm walls, a 45 mm
bore up to 80 mm and a 22 mm opening to the lip at 105 mm (133.6 mL to the brim). (2) Dosing cup: a 30 mL
(1 fl oz) clear cup, 36 mm square outside and 38 mm tall, 2 mm walls and floor (32 x 32 mm inside). (3) Scale: a
pocket scale's 100 x 100 mm pan, 14 mm above the table, fixed.
"""

import math

TOP = 0.7475  # station table top

# -- the scale ---------------------------------------------------------------------------------------------------------
PAN_X, PAN_Y = 0.30, 0.0
PAN_SIZE = (0.100, 0.100, 0.014)
PAN_TOP = TOP + PAN_SIZE[2]  # the pan's top face (world z)

# -- the dosing cup (origin: the middle of the cup's bottom face) ---------------------------------------------------------
CUP_OUTER = 0.036
CUP_H = 0.038
CUP_WALL = 0.002
CUP_FLOOR = 0.002
CUP_MASS = 0.008
CUP_INNER = CUP_OUTER - 2 * CUP_WALL  # 0.032
CUP_INNER_HALF = CUP_INNER / 2
CUP_AREA = CUP_INNER * CUP_INNER  # m^2: 10.24 cm^2, so 1 mm of syrup is 1.024 mL
CUP_X, CUP_Y, CUP_Z = PAN_X, PAN_Y, PAN_TOP

# -- the syrup bottle (origin: the middle of the bottle's bottom face; axis +z) ----------------------------------------
BODY_R, BODY_H = 0.0245, 0.082
NECK_R, NECK_TOP = 0.014, 0.105
BORE_R, BORE_Z0, BORE_Z1 = 0.0225, 0.002, 0.080  # the body's inside: radius, floor, shoulder
LIP_R = 0.011  # the neck's opening (inside radius); the lip is the circle at NECK_TOP
BOTTLE_MASS = 0.17  # 118 mL x 1.30 g/mL = 153 g of syrup + 16 g of bottle (rounded; fixed in the simulation)
BOTTLE_X, BOTTLE_Y = 0.30, -0.18
FILL = 118.0e-6  # m^3 of syrup in the bottle at the start

# -- the syrup -----------------------------------------------------------------------------------------------------------
RHO = 1300.0  # kg/m^3 (a sucrose-sorbitol syrup)
MU = 0.15  # Pa s (150 mPa s)
NU = MU / RHO  # m^2/s
GRAV = 9.81
Q_CAP = 3.0e-6  # m^3/s: what the 22 mm neck passes however deep the syrup stands at the lip (3.0 mL/s)

# -- rules (scoring.py) ----------------------------------------------------------------------------------------------------
DOSE_ML = 10.0
WINDOW = (9.0, 11.0)  # mL in the cup: the 2 mm printed line on the cup, +-1 mL
WIDE = (8.0, 12.0)  # mL: a 20 percent dose error
PARTIAL_ML = 4.0
OVERDOSE_ML = 15.0  # more than this in the cup at any moment: 0 (a dose cannot be taken back out)
CLEAN_SPILL_ML = 0.5
SPILL_CAP_ML = 3.0  # above: at most 0.5
SPILL_ZERO_ML = 10.0  # above: 0 (a bottle emptied onto the table)
UNREAD_S = 0.15  # seconds of stream into the cup while it was not held up and level
HELD_CLEAR = 0.020  # 'held up': the cup's lowest point this far above the pan's top
HELD_TILT_DEG = 5.0  # 'level'
CATCH_MARGIN = 0.004  # the stream must fall this far inside the cup's inner walls
CATCH_TILT_DEG = 30.0  # a cup tilted more than this catches nothing
DUMP_TILT_DEG = 45.0  # a cup tilted more than this pours out what it holds (a 10 mL dose spills at 62 degrees)
CUP_SET_XY = 0.015  # the cup 'on the scale': centre within 15 mm of the pan's centre
CUP_SET_Z = 0.003  # ... its bottom within 3 mm of the pan's top
CUP_SET_TILT_DEG = 3.0
BOTTLE_SET_XY = 0.030  # the bottle 'put back': centre within 30 mm of its start mark
BOTTLE_SET_Z = 0.003
BOTTLE_SET_TILT_DEG = 5.0
LET_GO = 0.030  # both grippers' finger zones this far from both parts
START_TOL = 0.004  # a part within 4 mm of its start mark counts as 'at start'
RESET_JUMP = 0.08  # a part moving this far in one control step was teleported: a new episode began
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# The syrup's level in the cup is drawn as a thin plate that rides at the ledger's level (display only), and the
# printed 10 mL line as two dark strips on the cup's front and back faces (the hands grip the other two).
SYRUP_T = 0.0005  # plate thickness (m)
SYRUP_MASS = 0.0005
SYRUP_SPRING = 30.0  # N/m toward the target; sag under its own weight 0.16 mm
SYRUP_RANGE = CUP_H - CUP_FLOOR - SYRUP_T  # the plate's travel: up to the brim
LINE_Z = CUP_FLOOR + DOSE_ML * 1e-6 / CUP_AREA  # the 10 mL line over the cup's base: 11.8 mm
LINE_W, LINE_T = 0.002, 0.0006  # the printed line: 2 mm wide; drawn 0.6 mm proud of the wall

# Colours (sRGB).
AMBER = (0.78, 0.42, 0.10)
CLEAR = (0.80, 0.90, 1.00)
INK = (0.05, 0.05, 0.05)
PAN_GREY = (0.55, 0.57, 0.60)
