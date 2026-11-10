"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A 2:1 bench mock-up of a pendulum impact tester's specimen area stands on the midline: a cast-iron base with two
supports and two anvils on it. The anvils are on the far side (+x); the striker would come from the near side (-x),
marked by a red strip on the base, and hit the specimen's back face at mid-span. A loaded specimen lies along y on
the supports, its notched face against both anvils (facing +x, away from the striker), the notch centred on the
midplane between the anvils (y 0).
"""

TOP = 0.7475  # station table top (world z)

# -- the tester base (fixed, cast iron) ------------------------------------------------------------------------------
BASE_X0, BASE_X1 = 0.270, 0.410  # 140 mm along x
BASE_HALF_Y = 0.110  # 220 mm along y
BASE_H = 0.030

# -- the two supports (fixed, hardened steel): the specimen lies on them ---------------------------------------------
SUP_X0, SUP_X1 = 0.300, 0.330  # 30 mm deep along x, toward the striker from the anvils' faces
SUP_Y_IN, SUP_Y_OUT = 0.040, 0.075  # each 35 mm along y, its inner edge 40 mm from the midplane
SUP_TOP = 0.040  # 10 mm tall on the base

# -- the two anvils (fixed, hardened steel): the specimen's notched face goes against them --------------------------
FACE_X = 0.330  # the anvils' faces toward the striker (= the supports' far edge)
ANVIL_X1 = 0.350  # 20 mm deep
ANVIL_Y_IN, ANVIL_Y_OUT = 0.040, 0.075  # inner edges 40 mm from the midplane: an 80 mm span (2 x the 40 mm standard)
ANVIL_TOP = 0.070  # 40 mm tall on the base: 30 mm above the supports' tops
MID_Y = 0.0  # the midplane between the anvils

# -- the striker mark (fixed red strip on the base, on the striker's side, in front of the gap) ----------------------
STRIKER_MARK_CENTER = (0.2825, 0.0)
STRIKER_MARK_SIZE = (0.025, 0.008, 0.001)

# -- the specimen: one loose rigid body (mechkit compound). Base frame: origin at the middle of its bottom face, local
#    +x along its length, local -y out of the notched face, local +z up through the black tick on its top face -------
SP_LEN, SP_W, SP_H = 0.110, 0.020, 0.020  # a 2:1 Charpy bar (the standard bar is 55 x 10 x 10 mm)
NOTCH_W, NOTCH_D = 0.004, 0.004  # a 4 mm wide, 4 mm deep slot across the notched face at mid-length, full height
SP_MASS = 0.118  # aluminium, 43.7 cm3
BACK_CENTER = (0.0, NOTCH_D / 2, SP_H / 2)  # the full-length 16 mm deep part behind the notch: y -0.006 .. +0.010
BACK_SIZE = (SP_LEN, SP_W - NOTCH_D, SP_H)
SIDE_LEN = (SP_LEN - NOTCH_W) / 2  # 53 mm: the two notch-side pieces either side of the slot
SIDE_SIZE = (SIDE_LEN, NOTCH_D, SP_H)
SIDE_CENTERS = ((-(NOTCH_W / 2 + SIDE_LEN / 2), -SP_W / 2 + NOTCH_D / 2, SP_H / 2),
                ((NOTCH_W / 2 + SIDE_LEN / 2), -SP_W / 2 + NOTCH_D / 2, SP_H / 2))
TICK_SIZE = (0.004, 0.006, 0.0005)  # a black tick on the top face, behind the notch
TICK_CENTER = (0.0, -SP_W / 2 + NOTCH_D + TICK_SIZE[1] / 2, SP_H + TICK_SIZE[2] / 2)  # y -0.006 .. 0.000
NOTCH_LOCAL = (0.0, -SP_W / 2, SP_H / 2)  # the centre of the notch's mouth on the notched face (mid-height)
NORMAL_LOCAL = (0.0, -1.0, 0.0)  # the notched face's outward normal
END_LOCAL = ((-SP_LEN / 2, 0.0, SP_H / 2), (SP_LEN / 2, 0.0, SP_H / 2))  # the centres of its two end faces
BOX_CENTER_LOCAL = (0.0, 0.0, (SP_H + TICK_SIZE[2]) / 2)  # its bounding box in its own frame (for the hands rule)
BOX_HALF = (SP_LEN / 2, SP_W / 2, (SP_H + TICK_SIZE[2]) / 2)

# -- the mat and where the specimen starts --------------------------------------------------------------------------
MAT_X, MAT_Y = 0.200, -0.220
MAT_SIZE = (0.140, 0.050, 0.003)
MAT_TOP = MAT_SIZE[2]
SP_START = (MAT_X, MAT_Y, MAT_TOP)  # the specimen's base-frame origin, lying along x, notched face toward -y
SP_START_YAW = 0.0

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
TILT_MAX_DEG = 5.0  # 'lying flat': the notch's 20 mm length (local z) within 5 degrees of vertical, either way up
HEADING_MAX_DEG = 10.0  # its length within 10 degrees of y (the line between the anvils) seen from above
SUPPORT_TOL = 0.002  # each end's underside within 2 mm of its support's top
ONTO_SUPPORT = 0.003  # each end at least 3 mm past its support's inner edge
NOTCH_CONE_DEG = 30.0  # 'notch away': the notched face's outward normal within 30 degrees of +x
CENTRE_TIGHT = 0.002  # +0.25: the notch within 2 mm of the midplane
CENTRE_LOOSE = 0.008  # beyond 8 mm: 0
ANVIL_GAP = 0.0015  # +0.25: the notched face within 1.5 mm of each anvil's face (a 1.5 mm feeler does not enter)
HANDS_OFF = 0.020  # every tier: both grippers' finger zones at least 20 mm from the specimen
STILL_WINDOW_S = 0.5  # 'at rest': neither end of the specimen moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the specimen moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- the grippers (the hands-off rule; rigstate reports the TCP, the point between the pad tips and the opening) -----
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
CAST_IRON = (0.24, 0.25, 0.27)
HARDENED_STEEL = (0.58, 0.60, 0.63)
STRIKER_RED = (0.85, 0.10, 0.08)
ALUMINIUM = (0.84, 0.85, 0.87)
TICK_BLACK = (0.04, 0.04, 0.04)
MAT_BLACK = (0.10, 0.10, 0.10)
