"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A wooden storage tray is fixed on the midline in front of the arms, its length along x. Two two-colour bar magnets
lie on the table, one on each side, along y with their red (north) ends pointing outward: the start is a mirror
image about the midline. Two soft-iron keepers lie nearer the arms, one on each side.
"""

TOP = 0.7475  # station table top (world z)

# -- the tray: a white-painted plywood floor with dark-stained beech rims on it (all fixed) ---------------------------
TRAY_X, TRAY_Y = 0.260, 0.000  # centre
INNER_X0, INNER_X1 = 0.175, 0.345  # inner faces of the end rims: 170 mm inside
INNER_Y = 0.063  # inner faces of the side rims at y +-0.063: 126 mm inside
RIM_T = 0.006  # rims 6 mm thick
RIM_H = 0.013  # and 13 mm tall above the floor
FLOOR_T = 0.003  # 3 mm plywood floor, as large as the tray's outside
FLOOR_SIZE = (INNER_X1 - INNER_X0 + 2 * RIM_T, 2 * (INNER_Y + RIM_T), FLOOR_T)  # 182 x 138 x 3
FLOOR_TOP = FLOOR_T  # the floor's top (above the table)
FLOOR_TOP_W = TOP + FLOOR_TOP
RIM_TOP = FLOOR_TOP + RIM_H  # 16 mm above the table

# -- the bar magnets: each one loose rigid body (mechkit compound) of two boxes end to end, base frame at the middle
#    of its bottom face, local +x toward the red (north) end, local +z up ----------------------------------------------
MAG_LEN, MAG_W, MAG_H = 0.100, 0.020, 0.015
MAG_MASS = 0.23  # a 100 x 20 x 15 mm steel bar is 0.235 kg
HALF_LEN = MAG_LEN / 2
NORTH_CENTER = (MAG_LEN / 4, 0.0, MAG_H / 2)  # the red half
SOUTH_CENTER = (-MAG_LEN / 4, 0.0, MAG_H / 2)  # the blue half
HALF_SIZE = (MAG_LEN / 2, MAG_W, MAG_H)
MAG_A_START = (0.220, 0.280, 0.0)  # left side: lying along y, red end toward +y (outward)
MAG_A_YAW = 1.5707963267948966
MAG_B_START = (0.220, -0.280, 0.0)  # right side: lying along y, red end toward -y (outward)
MAG_B_YAW = -1.5707963267948966

# -- the keepers: plain loose boxes, long axis along y -------------------------------------------------------------
KEEP_SIZE = (0.020, 0.074, 0.015)  # x, y, z as they start (and as they are laid across the pair)
KEEP_HALF_LEN = KEEP_SIZE[1] / 2
KEEP_MASS = 0.17  # soft iron, 20 x 74 x 15 mm
KEEP_1_START = (0.130, 0.220)  # left side, nearer the arms
KEEP_2_START = (0.130, -0.220)  # right side

# -- where the demo puts things (the scorer does not use these) ------------------------------------------------------
PAIR_X = TRAY_X  # magnets centred along the tray
PAIR_DY = 0.0225  # centres at y +-22.5 mm: a 25 mm gap between them (where a wooden spacer would go)
KEEP_GAP_AIM = 0.004  # each keeper set down 4 mm off the magnets' end faces
NEAR_END_X = PAIR_X - HALF_LEN  # 0.210
FAR_END_X = PAIR_X + HALF_LEN  # 0.310
KEEP_NEAR_X = NEAR_END_X - KEEP_GAP_AIM - KEEP_SIZE[0] / 2  # 0.196
KEEP_FAR_X = FAR_END_X + KEEP_GAP_AIM + KEEP_SIZE[0] / 2  # 0.324

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
STORE_LOW_TOL = 0.003  # 'on the floor': lowest point at most 3 mm above the tray floor
STORE_LEVEL_DEG = 5.0  # a stored magnet's long axis within 5 degrees of level
STORE_HEADING_DEG = 10.0  # ... and within 10 degrees of the tray's length (x) seen from above
RIM_SLACK = 0.001  # its whole outline seen from above inside the rims' inner faces (1 mm for contact)
LENGTHWISE_DEG = 30.0  # 'lengthwise in the tray' (the like-poles zero): long axis within 30 degrees of x
IN_TRAY_LOW_MAX = 0.020  # 'in the tray': centre inside the rims, lowest point under 20 mm above the floor (on the
#                          floor, or lying on the other magnet: 15 mm up)
PAIR_ACROSS_MAX = 0.060  # 'paired': centres at most 60 mm apart across the tray (y)
PAIR_ALONG_MAX = 0.015  # ... and within 15 mm of each other along it (x)
KEEP_LOW_TOL = 0.003  # a keeper 'on the floor': lowest point at most 3 mm above it
KEEP_LEVEL_DEG = 10.0  # ... its long axis within 10 degrees of level
KEEP_ACROSS_DEG = 15.0  # 'across the pair': its long axis within 15 degrees of y seen from above
KEEP_GAP_MIN = -0.002  # its near edge from each magnet's end face, along the tray: -2 mm (contact) ..
KEEP_GAP_MAX = 0.008  # .. 8 mm
KEEP_COVER_MIN = 0.012  # it spans at least 12 mm of each magnet's end face (20 mm wide; 15 mm on its side)
STILL_WINDOW_S = 0.5  # 'at rest': no point of the object moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # an object moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
NORTH_RED = (0.80, 0.08, 0.08)
SOUTH_BLUE = (0.10, 0.25, 0.78)
IRON = (0.22, 0.22, 0.24)
FLOOR_WHITE = (0.95, 0.95, 0.93)  # white-painted floor: the tray, the dark keepers and both magnet colours stand out
#                                    against it and against the light tan station table (a bare plywood floor would not)
RIM_STAIN = (0.42, 0.27, 0.15)  # dark-stained beech rims: a dark outline round the tray in the top view
