"""Every dimension of the task, in one place (metres, kilograms, rad, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A 12 fl oz (355 ml) round HDPE lotion pump bottle with a lock-down pump stands on a rubber shelf liner on the
vanity top, 300 mm in front of the arm bases and 250 mm to the right of the midline, in front of a tiled splash
wall. Its pump head is locked down on the closure by a bayonet (a quarter turn anticlockwise from above frees
it) and rides on a spring: free, it stands 23 mm higher, and pushed more than 4 mm down it makes a stroke.

Figures are listing figures (README) or estimates (est, said in task.json's fidelity).
"""

import math

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0

# -- the vanity top: shelf liner and splash wall (fixed) -------------------------------------------------------------
BX, BY = 0.280, -0.250  # the bottle's axis
LINER_X, LINER_Y, LINER_H = 0.220, 0.220, 0.003  # a 220 x 220 x 3 mm rubber shelf liner under the bottle (est)
LINER_FRICTION = 1.0
LINER_GREY = (0.30, 0.30, 0.32)
WALL_X0, WALL_X1 = 0.430, 0.450  # splash wall (tiled, 20 mm): 150 mm behind the bottle's axis, 100 mm behind the spout
WALL_Y0, WALL_Y1 = -0.620, -0.020
WALL_H = 0.300
WALL_TILE = (0.90, 0.92, 0.93)

# -- the bottle (loose) ----------------------------------------------------------------------------------------------
# 12 fl oz (355 ml) Boston round HDPE bottle with a 24/410 lock-down lotion pump. The sizes are design values inside
# the acceptance ranges of the README (the 16 fl oz retail figures there: 76 mm across, 193 mm locked, 216 mm free,
# 23 mm of travel, scale to this bottle); shoulder and closure sizes are estimates.
BODY_R = 0.033
BODY_H = 0.095
SHOULDER_R, SHOULDER_H = 0.028, 0.012  # to 107 mm
COLLAR_R, COLLAR_H = 0.0165, 0.015  # the 24/410 closure and its lug ring, to 122 mm
BASE_Z = LINER_H  # the bottle's base above the table top
HEAD_R = 0.016  # the ribbed head, 32 mm across
HEAD_H = 0.026
TRAVEL = 0.023  # 148 mm locked, 171 mm free
HEAD_UP = BODY_H + SHOULDER_H + COLLAR_H + TRAVEL  # 145 mm: the free head's underside above the bottle's base
HEAD_BOTTOM_LOCKED = HEAD_UP - TRAVEL  # 122 mm
HEAD_TOP_LOCKED = HEAD_BOTTOM_LOCKED + HEAD_H  # 148 mm
HEAD_TOP_UP = HEAD_UP + HEAD_H  # 171 mm
STEM_R = 0.006
SPOUT_R0, SPOUT_R1 = 0.010, 0.048  # from the axis to the nozzle's outer end
SPOUT_W, SPOUT_T = 0.009, 0.009
NOZZLE = 0.007  # the nozzle block under the spout's end, pointing down
NOZZLE_H = 0.008
SPOUT_BEARING0 = -90.0  # the spout points -y at the start (degrees, seen from above, 0 = +x)
BOTTLE_MASS = 0.42  # 355 ml of lotion (about 0.36 kg) in a 0.06 kg bottle and pump: a rigid mass
BASE_MASS, STEM_MASS, HEAD_MASS = 0.37, 0.02, 0.03
BOTTLE_FRICTION = 0.8
BOTTLE_WHITE = (0.93, 0.93, 0.90)
HEAD_BLACK = (0.10, 0.10, 0.11)
STEM_GREY = (0.70, 0.70, 0.72)

# the pump spring (slider "lift", axis -Z, 0 = free, TRAVEL = locked down)
K_LIFT = 400.0  # N/m: 1.6 N for the 4 mm stroke start, 9.2 N down at the lock (est)
D_LIFT = 9.0  # N s/m
# the bayonet (hinge "twist", axis +Z, positive = anticlockwise from above)
LUG_DEG = 80.0  # the lugs clear here (a quarter turn is 90)
LUG_RAD = math.radians(LUG_DEG)
K_TWIST = 0.025  # N m/rad: 0.035 N m at the lugs (est), standing for the bayonet
D_TWIST = 0.01  # N m s/rad: about the free swivel's 0.008 N m at 1 rad/s
TWIST_LIMIT = 3.4  # rad, either way

# -- scoring figures -------------------------------------------------------------------------------------------------
UNLOCK_UP = 0.018  # unlocked: the head 18 mm or more above its locked height
STROKE = 0.004  # a stroke: the head more than 4 mm below its top
SPOUT_TOL_DEG = 15.0  # the spout within 15 degrees of -x
STAND_TILT_DEG = 8.0
STAND_XY = 0.030  # base centre within 30 mm of the start
TIP_DEG = 45.0  # a tip: tilt over 45 degrees, latched
TURN_PART_DEG = 30.0  # the head turned 30 degrees toward unlocking
FALL_DROP = 0.050  # base more than 50 mm below its start height: off the vanity
LET_GO = 0.030  # finger zones 30 mm or more from the bottle
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # 5 mm and 5 degrees over 1 s: visible motion on a video
STILL_TURN_DEG = 5.0
RESET_JUMP = 0.05
RESET_LIFT = 0.015  # a head 15 mm further down in one control step: a reset, not a press
RESET_TWIST = 1.2  # rad in one control step
START_TOL = 0.01
EPS = 1e-9


def spout_bearing_deg(twist_rad: float, base_yaw_deg: float = 0.0) -> float:
    """The spout's bearing seen from above (0 = +x, anticlockwise positive) for a head turn and a bottle yaw."""
    return SPOUT_BEARING0 + math.degrees(twist_rad) + base_yaw_deg


def off_front_deg(bearing_deg: float) -> float:
    """The angle between a bearing and -x (toward the user), 0 to 180."""
    return abs((bearing_deg - 180.0 + 180.0) % 360.0 - 180.0)
