"""Every dimension of the task, in one place (metres, kilograms, radians, seconds unless named *_DEG or *_MM).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights here are above the table top unless named *_W.

A newborn training manikin lies on its back along x on the midline, crown toward the robot (the place the person
giving the breaths stands), its face up. In front of its crown a fixed tray holds an infant self-inflating bag with a
size-1 round face mask on its swivel elbow, mask down, bag pointing toward -y.

The bag-and-mask assembly's frame ("the mask frame", parts.py, scoring.py) has its origin at the centre of the
cushion's lower face (the pads uncompressed), z up the mask's axis, the valve housing and the bag toward -y.

Values marked (est) are estimates for the model, not published figures. Published: the mask's 64 mm outside
diameter and 37 g (Laerdal Newborn Mask size 1), the bag's 280 mL and 40 cmH2O relief valve (Intersurgical infant
bag, code 7150000). Everything else is an estimate.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
EPS = 1e-6
START_TOL = 0.010  # the bag and mask "on their marks" (reset detection)
RESET_JUMP = 0.050  # a jump no trial makes in one step
STILL_WINDOW_S = 1.0

HEAD, BODY, TRAY, BVM = "manikin_head", "manikin_body", "tray", "bag_mask"

# -- the manikin (fixed; est: a newborn trainer about 50 cm long) ------------------------------------------------------
SKIN = (0.93, 0.78, 0.68)
EYE_DARK = (0.20, 0.16, 0.14)
LIP = (0.72, 0.42, 0.40)
FACE_X, FACE_Y = 0.250, 0.000  # the mask's target point: midway between the tip of the nose and the chin
FACE_Z = 0.090  # the face under the mask: a level rigid seat (est)
FACE_R = 0.042  # that seat's radius: 84 mm across (est)
FACE_Z0 = 0.050  # its lower end (inside the skull)
SKULL_X = FACE_X - 0.0235  # the skull's centre (x)
SKULL_HALF = 0.0625  # its half-length along x: crown at x 164 mm, chin at x 289 mm (est)
SKULL_R = 0.044  # its radius: 88 mm across (est)
SKULL_AXIS_Z = 0.046
SKULL_SLICES = 9
NOSE = ((-0.014, -0.006), 0.005, 0.003)  # x span from the target, half-width, height above the face (est)
MOUTH = ((0.008, 0.013), 0.007, 0.0005)
EYE_DX, EYE_DY = -0.044, 0.021  # each eye's centre from the target point (est)
EYE_DISC_R = 0.006
EYE_R = 0.0075  # the eye zone: a 15 mm ball round each eye's centre
NECK_X = (0.293, 0.310)
NECK_R, NECK_AXIS_Z = 0.022, 0.040
TORSO_X = (0.310, 0.460)
TORSO_R, TORSO_AXIS_Z = 0.050, 0.052
CHEST_X, CHEST_SIZE = 0.365, (0.080, 0.060, 0.006)  # the chest plate on the torso's top, on a vertical slide
CHEST_Z0 = 0.098  # its lower face at rest; its top is at 104 mm
CHEST_TRAVEL = 0.015
CHEST_K = 2000.0  # N/m (est): stiff, so the declared lung rule sets the rise
CHEST_MASS = 0.05
CHEST_DAMP = 20.0
ARM_X, ARM_Y, ARM_R, ARM_AXIS_Z = (0.320, 0.420), 0.066, 0.016, 0.018
LEG_X, LEG_Y, LEG_R, LEG_AXIS_Z = (0.460, 0.580), 0.028, 0.020, 0.022
FACE_FRICTION = (0.7, 0.6)  # silicone on the manikin's vinyl skin (est)


def eye_skull_z(dx: float = EYE_DX, dy: float = EYE_DY) -> float:
    """Height of the skull's surface at (FACE_X + dx, dy): where each eye sits."""
    u = (FACE_X + dx - SKULL_X) / SKULL_HALF
    r = SKULL_R * math.sqrt(max(0.0, 1.0 - u * u))
    return SKULL_AXIS_Z + math.sqrt(max(0.0, r * r - dy * dy))


EYE_Z = eye_skull_z()
EYES = ((FACE_X + EYE_DX, FACE_Y + EYE_DY, EYE_Z), (FACE_X + EYE_DX, FACE_Y - EYE_DY, EYE_Z))

# -- the tray (fixed), in front of the crown -------------------------------------------------------------------------
TRAY_X = (0.062, 0.158)
TRAY_Y = (-0.245, 0.045)
TRAY_Z = 0.010  # its top
TRAY_COLOR = (0.82, 0.84, 0.86)

# -- the bag and mask (one loose articulation; the mask frame) -------------------------------------------------------
BVM_X0, BVM_Y0 = 0.110, 0.000  # where the mask's axis stands on the tray
CLEAR_SILICONE = (0.80, 0.86, 0.90)
VALVE_BLUE = (0.35, 0.55, 0.85)
BAG_GREEN = (0.55, 0.78, 0.62)
# the mask: published 64 mm outside diameter and 37 g; the rest est
MASK_OD = 0.064
MASK_MASS = 0.037
PAD_N = 8  # the soft cushion: 8 pads on vertical sprung slides (est)
PAD_RC = 0.0275
PAD_RADIAL = 0.008  # 23.5 to 31.5 mm from the axis
PAD_TAN = 0.020
PAD_H = 0.010
PAD_TRAVEL = 0.010
PAD_K = 150.0  # N/m each, 1200 N/m for the cushion (est)
PAD_MASS = 0.003
PAD_DAMP = 1.5
MASK_ROOT_MASS = MASK_MASS - PAD_N * PAD_MASS
RIM_IN, RIM_OUT = 0.0235, MASK_OD / 2
RIM_Z = (PAD_H, 0.015)
DOME_R, DOME_Z = 0.027, (0.015, 0.018)
TOP_R, TOP_Z = 0.018, (0.018, 0.0195)
# the valve: elbow on a 360 degree swivel, housing toward -y (est)
SWIVEL_Z = TOP_Z[1]
ELBOW_R, ELBOW_TOP = 0.0095, 0.058
HOUSING_Z = 0.029  # the housing's axis
HOUSING_Y = (-0.004, -0.060)
VALVE_MASS = 0.030
SWIVEL_LIMIT = 3.0
SWIVEL_DAMP = 0.002
# the bag: 280 mL published; 150 mm long and 56 mm across est (a 280 mL ellipsoid is about 60 mm across)
NECK_Y = HOUSING_Y[1]
NECK_LEN, NECK_RAD = 0.013, 0.010
BAG_Y = (NECK_Y - NECK_LEN, NECK_Y - NECK_LEN - 0.150)  # -0.073 to -0.223
BAG_R = 0.028
BAG_END_R = 0.012
BAG_SLICES = 9
INTAKE_LEN, INTAKE_R = 0.014, 0.008
BAG_MASS = 0.070
NECK_LIMIT = 0.5  # the neck bends +-0.5 rad, freely (est: a soft silicone neck)
NECK_KINK = 0.40  # bent further, it is kinked: no breath goes through (declared rule)
NECK_DAMP = 0.05
PANEL_Y = (BAG_Y[0] + BAG_Y[1]) / 2  # -0.148: the bag's middle
PANEL_LEN, PANEL_H, PANEL_T = 0.060, 0.044, 0.007
PANEL_OUT = 0.037  # outer faces 74 mm apart
BAG_ACROSS = 2 * PANEL_OUT
PANEL_TRAVEL = 0.012
PANEL_K = 400.0  # N/m each (est): 10 mm of squeeze (5 per side) takes 2 N a side
PANEL_MASS = 0.005
PANEL_DAMP = 2.0
BVM_MASS = MASK_MASS + VALVE_MASS + BAG_MASS + 2 * PANEL_MASS
BVM_FRICTION = (0.7, 0.6)  # est

# -- the gripper (yam_arena: parallel pads, 0 to 93.9 mm; each finger 5000 N/m toward its target, capped at 50 N) ---
FULL_GAP = 0.0939
FINGER_K = 5000.0
FINGER_CAP = 50.0
FINGER_SHARE = FINGER_K / (FINGER_K + PANEL_K)  # of a commanded squeeze that reaches a panel
VALVE_W = 2 * ELBOW_R  # 19 mm across the elbow
VALVE_GRIP_Z = 0.0535  # the left's TCP over the mask origin: pad tips 3 mm over the housing, on the elbow's top 17 mm

# -- the declared air rule (est; the bag, mask and lungs as a rule, read in "rise" units) -------------------------------
LUNG_GAIN = 0.6  # mm of chest rise per mm of further squeeze while sealed
EASE = 0.001  # easing the squeeze 1 mm from its peak opens the valve: the lungs breathe out
TAU_OUT = 0.3  # s: passive breath out
TAU_LEAK = 0.2  # s: a leak (no seal) or a kinked neck: the chest falls back
RISE_CAP = 0.012  # the 40 cmH2O relief valve caps the rise (est)

# -- the seal (declared: the cushion pressed evenly on the face) -------------------------------------------------------
SEAL_LAT = 0.008  # the mask's axis within 8 mm of the target point (top view)
SEAL_TILT_DEG = 6.0
SEAL_HIGH = 0.0005  # the cushion's lower face no more than 0.5 mm over the face (touching)
PAD_MIN = 0.0003  # every pad pressed 0.3 mm or more
PAD_STOP = 0.0095  # and none at its 10 mm stop
SEAL_F = (2.0, 10.0)  # N: the cushion's total spring force; over 10 N presses the airway shut (est)

# -- scoring ---------------------------------------------------------------------------------------------------------
RISE_OK = 0.004  # an inflation: the chest 4 mm or more up
RISE_LOW = 0.001  # it has fallen back
RISE_OVER = 0.009  # over-inflated
BREATH = (2.0, 3.0)  # s: a good inflation's length
N_BREATHS = 5
LIFT_OFF_S = 1.0  # after the first good inflation, unsealed for longer than this: "lifted"
EYE_TOUCH = 0.001  # the bag and mask within 1 mm of an eye zone
EYE_HAND = 0.002  # a gripper's finger zone within 2 mm of an eye zone
DROP_LIFT = 0.010  # lifted off the tray
DROP_TABLE = 0.002  # then within 2 mm of the table
REST_SQUEEZE = 0.002  # the bag let go: squeezed under 2 mm
BAG_LET_GO = 0.005  # every finger zone 5 mm or more from the bag
