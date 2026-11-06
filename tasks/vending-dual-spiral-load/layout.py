"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left of the midpoint between them, z up.
The arm bases stand at x 0, y +-0.305; the station's table top is at z 0.7475.

The scene is one vending-machine spiral tray (two neighbouring spirals) fixed in front of the left arm, a loose lane
divider standing in the slot between the two spirals, a grey parts mat on the midline and four wide product boxes
standing on the table to the left of the tray.
"""

TOP = 0.7475  # station table top

# -- the spiral tray: a fixed steel plate; its top is the shelf floor ------------------------------------------------
TRAY_X0, TRAY_X1 = 0.160, 0.420  # x 0.160 is the front (delivery) edge, facing the arms
TRAY_Y0, TRAY_Y1 = 0.040, 0.240
TRAY_H = 0.020
TRAY_SIZE = (TRAY_X1 - TRAY_X0, TRAY_Y1 - TRAY_Y0, TRAY_H)  # 260 x 200 x 20 mm
TRAY_CENTRE = ((TRAY_X0 + TRAY_X1) / 2, (TRAY_Y0 + TRAY_Y1) / 2)  # (0.290, 0.140)
SHELF = TOP + TRAY_H  # the shelf floor, 20 mm above the table

# -- two side walls: fixed steel bars along x on the tray ----------------------------------------------------------
WALL_W, WALL_H = 0.010, 0.030
WALL_SIZE = (TRAY_SIZE[0], WALL_W, WALL_H)  # 260 x 10 x 30 mm
WALL_INNER_R, WALL_INNER_L = 0.050, 0.230  # inner faces: the right wall spans y 0.040-0.050, the left 0.230-0.240
WALLS = (("wall_right", WALL_INNER_R - WALL_W / 2), ("wall_left", WALL_INNER_L + WALL_W / 2))
WALL_TOP = SHELF + WALL_H  # 50 mm above the table

# -- the two coils, as fixed ribs where their loops meet the shelf -------------------------------------------------
RIB_T, RIB_H = 0.008, 0.012  # 8 mm along x, 12 mm tall
RIB_XS = (0.164, 0.226, 0.288, 0.350, 0.412)  # rib centres: 62 mm pitch, lined up across both spirals
RIB_SPANS = {"b": (0.050, 0.128), "a": (0.152, 0.230)}  # spiral B (right) and spiral A (left), 78 mm each along y
RIB_TOP = SHELF + RIB_H  # 32 mm above the table
SLOT_Y0, SLOT_Y1 = RIB_SPANS["b"][1], RIB_SPANS["a"][0]  # the 24 mm slot between the spirals, no ribs
LANE_Y = (SLOT_Y0 + SLOT_Y1) / 2  # 0.140: the selection's centreline
# The four pitches: the clear spaces between rib faces, 54 mm each along x.
PITCHES = tuple((RIB_XS[k] + RIB_T / 2, RIB_XS[k + 1] - RIB_T / 2) for k in range(4))
PITCH_X = tuple((a + b) / 2 for a, b in PITCHES)  # 0.195, 0.257, 0.319, 0.381

# -- the selection: over the shelf between the walls' inner faces ---------------------------------------------------
SEL_X0, SEL_X1 = TRAY_X0, TRAY_X1
SEL_Y0, SEL_Y1 = WALL_INNER_R, WALL_INNER_L
SEL_INSET = 0.001  # 'inside' means more than 1 mm inside that area (a part touching the tray's end faces is not)

# -- the lane divider: a loose white HDPE bar standing on the shelf in the slot -------------------------------------
DIV_SIZE = (0.230, 0.020, 0.040)
DIV_MASS = 0.175  # HDPE (0.95 g/cm3) of this size: 175 g
DIV_X, DIV_Y = 0.285, LANE_Y  # x 0.170-0.400, y 0.130-0.150: 2 mm clear of the ribs each side

# -- the parts mat: a fixed grey rubber sheet on the midline --------------------------------------------------------
MAT_SIZE = (0.260, 0.060, 0.005)
MAT_X, MAT_Y = 0.285, -0.020  # x 0.155-0.415, y -0.050..+0.010
MAT_TOP = TOP + MAT_SIZE[2]

# -- four wide product boxes: loose cartons standing on the table, 140 mm side along y ------------------------------
BOX_SIZE = (0.030, 0.140, 0.070)  # 30 mm along x, 140 mm along y, 70 mm tall
BOX_MASS = 0.090
BOX_Y = 0.340  # y 0.270-0.410
BOX_XS = (0.180, 0.260, 0.340, 0.420)  # 50 mm gaps; the box at x 0.180 goes to pitch 1, and so on
BOX_NAMES = ("box_1", "box_2", "box_3", "box_4")

# -- rule thresholds (scoring.py) -----------------------------------------------------------------------------------
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
STILL_S = 0.5  # 'at rest': the centre's path over the last 0.5 s ...
STILL_PATH = 0.002  # ... is under 2 mm
HELD_OPENING = 0.5  # 'held': a gripper less than half open (pads under 47 mm apart) ...
HELD_GROW = 0.015  # ... with its pad-tip point inside the object's outline grown by 15 mm on every side
# Seated: the box's lowest point 19 to 23 mm above the table top (on the 20 mm shelf floor: up to 3 mm above it, and
# 1 mm below it for a resting contact that reads slightly into its support).
CONTACT_TOL = 0.001
SEAT_LOW_MIN, SEAT_LOW_MAX = SHELF - CONTACT_TOL, SHELF + 0.003
SEAT_TILT_DEG = 8.0  # ... its 70 mm axis within 8 degrees of vertical
SEAT_Y_TOL = 0.020  # ... its centre within 20 mm of the lane centreline (y 0.140)
SEAT_YAW_DEG = 10.0  # ... its 140 mm side within 10 degrees of the y axis
# Divider on the mat: its lowest point 4 to 10 mm above the table top (on the mat's 5 mm top: up to 5 mm above it,
# 1 mm below it for contact; the bare table, 5 mm below the mat's top, is out).
MAT_LOW_MIN, MAT_LOW_MAX = MAT_TOP - CONTACT_TOL, MAT_TOP + 0.005
RESET_JUMP = 0.05  # an object moving this far in one control step was teleported ...
START_TOL = 0.045  # ... onto its starting mark (within this of it; covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
STEEL_GREY = (0.72, 0.73, 0.74)
COIL_BLACK = (0.06, 0.06, 0.07)
HDPE_WHITE = (0.94, 0.94, 0.92)
MAT_GREY = (0.32, 0.33, 0.34)
CARTON_RED = (0.82, 0.22, 0.14)  # the four cartons are the same product, printed alike
