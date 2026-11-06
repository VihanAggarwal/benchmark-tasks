"""Every dimension and threshold of the task, in one place (metres, kilograms, seconds; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say
"world". Yaw is anticlockwise seen from above.
"""

import math

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the drive line --------------------------------------------------------------------------------------------
SHAFT_Z = 0.052  # both shafts' axes, 52 mm above the table (level by construction)
PLANE_X = 0.230  # the blower pulley's outer face (toward the arms): the line the motor's pulley face must reach
CENTRE_DIST = 0.290  # pulley centre to pulley centre along y (blower y -0.150, motor y +0.140)

# -- the blower: one FIXED mechkit compound (no mass), its frame on the table under the pulley's centre ----------
BLOWER_ORIGIN = (0.240, -0.150, 0.0)
BLOWER_HOUSING_C, BLOWER_HOUSING = (0.090, 0.0, 0.070), (0.120, 0.100, 0.140)  # world x 0.270-0.390, y -0.200..-0.100
BLOWER_SHAFT_C, BLOWER_SHAFT_R, BLOWER_SHAFT_LEN = (0.020, 0.0, SHAFT_Z), 0.008, 0.020  # x 0.250-0.270, along x
BLOWER_PULLEY_C, BLOWER_PULLEY_R, BLOWER_PULLEY_LEN = (0.0, 0.0, SHAFT_Z), 0.045, 0.020  # x 0.230-0.250, 90 mm across
BLOWER_Y = BLOWER_ORIGIN[1]
OUTER_RIM_Y = BLOWER_Y - BLOWER_PULLEY_R  # -0.195: the blower pulley's rim away from the motor
INNER_RIM_Y = BLOWER_Y + BLOWER_PULLEY_R  # -0.105: its rim toward the motor

# -- the motor's base plate: a FIXED steel plate -----------------------------------------------------------------
PLATE_SIZE = (0.160, 0.130, 0.012)
PLATE_C = (0.342, 0.140)  # outline x 0.262-0.422, y 0.075-0.205; its front edge 32 mm behind the line
PLATE_TOP = PLATE_SIZE[2]  # 0.012
PLATE_X0, PLATE_X1 = PLATE_C[0] - PLATE_SIZE[0] / 2, PLATE_C[0] + PLATE_SIZE[0] / 2
PLATE_Y0, PLATE_Y1 = PLATE_C[1] - PLATE_SIZE[1] / 2, PLATE_C[1] + PLATE_SIZE[1] / 2

# -- the motor: one LOOSE mechkit compound (a bench model with a printed shell) ----------------------------------
# Its frame: origin at the centre of the body's underside, local x along the body, the shaft out of local -x.
MOTOR_MASS = 0.15  # spread evenly over its shapes' volume (as PhysX does): centre of mass 8.6 mm from the body's
#                   centre toward the pulley, 40 mm up, so a push on its pulley face more than 45 mm above its underside
#                   (static friction 1.2) would tip it rather than slide it; the straightedge's top is 34 mm up
MOTOR_BODY_C, MOTOR_BODY = (0.0, 0.0, 0.040), (0.090, 0.070, 0.080)
MOTOR_SHAFT_C, MOTOR_SHAFT_R, MOTOR_SHAFT_LEN = (-0.0575, 0.0, 0.040), 0.008, 0.025  # local x -0.070..-0.045
MOTOR_PULLEY_C, MOTOR_PULLEY_R, MOTOR_PULLEY_LEN = (-0.079, 0.0, 0.040), 0.032, 0.018  # local x -0.088..-0.070
FACE_LOCAL = (-0.088, 0.0, 0.040)  # the centre of the motor pulley's outer face, in the motor's frame
MOTOR_Y = 0.140
MOTOR_START = (0.330, MOTOR_Y, PLATE_TOP)  # origin, sitting on the plate
MOTOR_YAW0_DEG = 3.0
MOTOR_YAW0 = math.radians(MOTOR_YAW0_DEG)
ALIGNED_X = PLANE_X - FACE_LOCAL[0]  # 0.318: the origin's x with the face on the line at yaw 0
BODY_HALF = (MOTOR_BODY[0] / 2, MOTOR_BODY[1] / 2)
# The box around every motor shape (what ObjectState.pos is the centre of at rest), in the motor's frame.
MOTOR_BOX_CENTRE = (-0.0215, 0.0, 0.040)
# At the start the face centre is at x 0.2421: 12.1 mm behind the line, turned 3 deg: M = 5.39 deg.

# -- the straightedge: a LOOSE aluminium box section standing on its 20 mm edge on a felt strip ------------------
BAR_SIZE = (0.020, 0.480, 0.040)
BAR_HALF = tuple(s / 2 for s in BAR_SIZE)
BAR_MASS = 0.18  # 20 x 40 mm, 1.2 mm wall, 480 mm long
FELT_SIZE = (0.040, 0.500, 0.002)
FELT_C = (0.150, -0.045)
BAR_START = (FELT_C[0], FELT_C[1], FELT_SIZE[2] + BAR_SIZE[2] / 2)  # centre: spans y -0.285 .. +0.195

# -- scoring -------------------------------------------------------------------------------------------------
# Total misalignment M = |theta| + alpha (degrees), angular plus parallel, as the belt maker's guidance adds them;
# alpha = atan(|e| / CENTRE_DIST).
TIER_FULL_DEG = 0.5  # 1.0: the V-belt limit (offset alone 2.53 mm, or angle alone 0.5 deg)
TIER_HALF_DEG = 1.0  # 0.5 (offset alone 5.06 mm)
TIER_QUARTER_DEG = 2.0  # 0.25 (offset alone 10.1 mm)
UPRIGHT_DEG = 2.0  # the body's up axis within this of vertical
PLATE_Z_TOL = 0.002  # every bottom corner of the body within this of the plate's top
Y_TOL = 0.010  # the motor's origin within this of y 0.140 (keeps the 290 mm centre distance that alpha uses)
HANDS_OFF = 0.060  # every TCP and pad-tip point at least this far from the motor's and the straightedge's boxes:
#                   more than a fully open finger reaches (its outer face FULL_GAP / 2 + FINGER_THICK = 57 mm out)
STILL_STEPS = 10  # 'at rest' is judged over the last 0.5 s (10 control steps) ...
STILL_PATH = 0.002  # ... the face centre's path over that window shorter than this
STILL_TURN_DEG = 0.5  # ... and the motor's heading turned less than this in all
RESET_JUMP = 0.025  # an object moving this far in one control step (0.5 m/s) was put there by the simulator ...
START_TOL = 0.045  # ... and, if every loose object is then within this of its start, a new episode began
START_YAW_TOL_DEG = 10.0  # ... turned within this of its starting heading

# -- the grippers (yam_arena) ----------------------------------------------------------------------------------
FULL_GAP = 0.0939  # pads 94 mm apart fully open; rigstate reports the opening 0 (closed) .. 1 (open)
FINGER_THICK = 0.010  # each finger's thickness outside the gap (19-38 mm up from its tip)

# -- offline model only (offline_model.py; the simulator never uses these) ------------------------------------
HOLD_GAP = BAR_SIZE[0] + 0.010  # a gripper at most 30 mm open ...
HOLD_REACH = 0.015  # ... with its pad-tip point within 15 mm of the straightedge holds it

# -- colours (sRGB) ------------------------------------------------------------------------------------------
STEEL_DARK = (0.28, 0.29, 0.31)
STEEL = (0.62, 0.63, 0.65)
CAST_IRON = (0.08, 0.08, 0.09)
GALVANISED = (0.72, 0.74, 0.76)
MOTOR_BLUE = (0.12, 0.30, 0.62)
ALUMINIUM = (0.82, 0.83, 0.85)
FELT_GREEN = (0.10, 0.42, 0.22)
