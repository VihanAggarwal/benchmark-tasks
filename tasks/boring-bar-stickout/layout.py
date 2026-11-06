"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

The lathe is a bench-scale mock with its spindle axis along y at x 0.250, 48 mm above the table: the chuck and the
workpiece on the left (+y), the tool block on the midline facing them. The boring bar points along +y (toward the
chuck) when it is set, so its stick-out is read along y from the block's chuck-side face.
"""

TOP = 0.7475  # station table top (world z)

# -- the spindle axis (the bore's axis) ------------------------------------------------------------------------------
AXIS_X = 0.250

# -- the tool block: a fixed base with two walls on top forming a channel along y ----------------------------------
BLOCK_X0, BLOCK_X1 = 0.225, 0.275  # 50 mm across
BLOCK_Y0, FACE_Y = -0.080, 0.000  # 80 mm long; FACE_Y is its chuck-side face (the stick-out is measured from it)
FLOOR_Z = 0.040  # the base's top = the channel floor
WALL_T = 0.012  # each wall 12 mm thick
WALL_H = 0.012  # and 12 mm tall
WALL_TOP = FLOOR_Z + WALL_H  # 0.052
CHANNEL_X0, CHANNEL_X1 = BLOCK_X0 + WALL_T, BLOCK_X1 - WALL_T  # 0.237 .. 0.263: a 26 mm channel
CHANNEL_W = CHANNEL_X1 - CHANNEL_X0

# -- the boring bar: one loose rigid body (mechkit compound), base frame at the middle of its bottom face, local +x
#    toward the cutting tip, local +z up through the insert ----------------------------------------------------------
BAR_W = 0.016  # 16 x 16 mm square shank
BAR_LEN = 0.150
HEAD_LEN = 0.015  # the black-oxide head at the tip end
SHANK_LEN = BAR_LEN - HEAD_LEN
INSERT_SIZE = (0.006, 0.006, 0.003)  # the gold insert on top of the head, flush with the tip face
BAR_MASS = 0.30  # a 16 x 16 x 150 mm steel bar is 0.30 kg
SHANK_CENTER = (-BAR_LEN / 2 + SHANK_LEN / 2, 0.0, BAR_W / 2)  # (-0.0075, 0, 0.008)
SHANK_SIZE = (SHANK_LEN, BAR_W, BAR_W)
HEAD_CENTER = (BAR_LEN / 2 - HEAD_LEN / 2, 0.0, BAR_W / 2)  # (0.0675, 0, 0.008)
HEAD_SIZE = (HEAD_LEN, BAR_W, BAR_W)
INSERT_CENTER = (BAR_LEN / 2 - INSERT_SIZE[0] / 2, 0.0, BAR_W + INSERT_SIZE[2] / 2)  # (0.072, 0, 0.0175)
TIP_LOCAL = (BAR_LEN / 2, 0.0, BAR_W / 2)  # centre of the tip face
BACK_LOCAL = (-BAR_LEN / 2, 0.0, BAR_W / 2)  # centre of the back end's face
# Mass is spread over the colliders by volume, so the centre of mass is 0.2 mm toward the tip from the middle (the
# insert's 108 mm3): a bar lying in the channel balances on the block's face edge at a stick-out of about 74.8 mm.
BALANCE_STICKOUT = 0.0748

# -- the chuck, the workpiece and the bore mark (all fixed) ----------------------------------------------------------
AXIS_Z = FLOOR_Z + BAR_W / 2  # 0.048: the bore's axis height = a seated bar's centre line
WORK_FACE_Y = FACE_Y + 0.075  # the workpiece's face toward the tool: a tip there has a stick-out of 75 mm
WORK_SIZE = (0.050, 0.040, 0.050)  # x, y, z: a 50 mm square aluminium block, 40 mm out of the chuck
WORK_CENTER = (AXIS_X, WORK_FACE_Y + WORK_SIZE[1] / 2, AXIS_Z)
CHUCK_SIZE = (0.100, 0.030, 0.096)
CHUCK_CENTER = (AXIS_X, WORK_FACE_Y + WORK_SIZE[1] + CHUCK_SIZE[1] / 2, CHUCK_SIZE[2] / 2)  # y 0.115 .. 0.145
MARK_SIZE = (0.022, 0.0005, 0.022)  # a black 22 mm square, 0.5 mm thick, on the workpiece face
MARK_CENTER = (AXIS_X, WORK_FACE_Y - MARK_SIZE[1] / 2, AXIS_Z)

# -- the tool mat and where the bar starts ---------------------------------------------------------------------------
MAT_SIZE = (0.180, 0.060, 0.003)
MAT_X, MAT_Y = 0.195, -0.170
MAT_TOP = MAT_SIZE[2]
BAR_START = (MAT_X, MAT_Y, MAT_TOP)  # the bar's base-frame origin (middle of its bottom face), lying along x
BAR_START_YAW = 0.0  # tip toward +x (away from the arms), insert up

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
BAND_LO, BAND_HI = 0.045, 0.064  # 1.0: bore depth 40 mm + 5 mm .. four times the 16 mm shank
SHORT_LO = 0.038  # 0.5 from 38 mm up to 45 mm (short of the full depth)
LONG_HI = 0.072  # 0.5 above 64 mm up to 72 mm (past the four-times rule); above 72 mm the tip is within 3 mm of
#                  the workpiece face (2.5 mm of the bore mark): 0
CENTRE_TOL = (CHANNEL_W - BAR_W) / 2 + 0.001  # 6 mm: the centre line's offset from the channel's centre at either
#                                               end of the block (5 mm of play, 1 mm for the solver); more: on a wall
FLOOR_TOL = 0.003  # the bar's underside within 3 mm of the channel floor at either end of the block
TILT_MAX_DEG = 5.0  # the insert-up axis within 5 degrees of vertical (covers pitch, roll and a bar on its side)
HEADING_MAX_DEG = 10.0  # the bar's axis within 10 degrees of y seen from above (in the channel it cannot exceed 7)
MIN_OVER_BLOCK = 0.020  # at least 20 mm of the bar's length over the block
STILL_WINDOW_S = 0.5  # 'at rest': neither end of the bar moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the bar moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
STEEL = (0.62, 0.64, 0.67)  # the bar's shank
HEAD_BLACK = (0.12, 0.12, 0.13)  # its black-oxide head
GOLD = (0.90, 0.68, 0.15)  # the insert (TiN-coated carbide)
BLOCK_STEEL = (0.30, 0.33, 0.38)  # the tool block
CHUCK_GREY = (0.52, 0.53, 0.55)
ALUMINIUM = (0.80, 0.81, 0.83)
MARK_BLACK = (0.05, 0.05, 0.05)
MAT_BLACK = (0.10, 0.10, 0.10)
