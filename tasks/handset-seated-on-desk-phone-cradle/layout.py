"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top (z 0.7475) is the desk top.

Scene: a desk phone (a loose 1.10 kg base with two cup recesses in its top and a spring hook-switch plunger between
them) stands on the desk at x 0.30, y 0. A handset lies on the desk to its right, face down on its two end pads, long
axis along x, its earpiece end pointing away from the robot. A desktop monitor stands behind the phone and a pen pot
beside it. The handset has to be turned a quarter turn, lowered into the cups (earpiece end into the narrower cup)
and let go; it then presses the hook switch down, which is what ends the call.

Frames. The phone's frame sits at the centre of its underside (z up). The handset's frame has the same x, y axes as the
phone's when it is seated, and its origin at the centre of the pad-bottom plane over the cradle's centre, so a seated
handset's frame coincides (x, y) with the phone's, 45 mm up (the cup floor). Heights named *_Z are above that origin
(the underside of the part).
"""

import math

TOP = 0.7475  # station table top = the desk top

# -- the phone base (loose, 1.10 kg): rubber feet on the desk, two cups in its top, the hook switch between them ------
PHONE_X, PHONE_Y = 0.30, 0.0  # centre of its underside
PHONE_SIZE = (0.200, 0.240)  # along x (front to back), along y
PHONE_H = 0.055  # its top
CUP_DEPTH = 0.010  # both cups
FLOOR_Z = PHONE_H - CUP_DEPTH  # 0.045: the cups' floor above the phone's underside
PHONE_MASS = 1.10
PHONE_FRICTION = (1.0, 0.9)  # rubber feet on a laminate desk (est); the handset and the grippers average with it
PHONE_CHARCOAL = (0.16, 0.16, 0.18)

# The two cups: centre (x, y) in the phone's frame and inside size (x, y). Ear cup 58 mm long (y), mouth cup 82 mm long.
EAR_CUP = ((0.0, 0.072), (0.060, 0.058))
MOUTH_CUP = ((0.0, -0.067), (0.068, 0.082))
CUPS = {"ear": EAR_CUP, "mouth": MOUTH_CUP}

# The hook switch: a round plunger between the cups, 8 mm proud of the top, pushed down by the handset's handle.
HOOK = "hook"
HOOK_XY = (0.0, 0.008)  # in the phone's frame: the middle of the flat top between the cups (y -0.026 to +0.043)
HOOK_RADIUS = 0.007
HOOK_PROUD = 0.008  # its top stands this far over the phone's top (free); it can go down this far (flush)
HOOK_TRAVEL = HOOK_PROUD
HOOK_SPRING = 200.0  # N/m: 0.8 N at the 4 mm a seated handset presses it, less than the handset's 1.67 N weight
HOOK_DAMPING = 2.0
HOOK_MASS = 0.02
HOOK_FREE_TOP = PHONE_H + HOOK_PROUD  # 0.063
HOOK_BODY = 0.023  # the plunger's shaft, from 40 mm up to its top
HOOK_RED = (0.55, 0.12, 0.10)

# -- the handset (loose, 0.17 kg): two end pads joined by an arched handle --------------------------------------------
# Pads (the cups' plugs): earpiece 46 mm (y, long axis) by 44 mm (x) and mouthpiece 70 mm by 52 mm, both 28 mm tall,
# centred 72 mm and 67 mm either side of the cradle's centre line: 8 mm clear each side across (x) and 6 mm each end
# (y) in every cup.
EAR_PAD = ((0.0, 0.072), (0.044, 0.046), 0.028)  # centre (x, y), size (x, y), height
MOUTH_PAD = ((0.0, -0.067), (0.052, 0.070), 0.028)
PADS = {"ear": EAR_PAD, "mouth": MOUTH_PAD}
# Handle: 34 mm wide (x), 34 mm tall, from y -70 to +60 mm, its underside 14 mm over the pad bottoms (the arch),
# so the pads hang 14 mm below it and its top is 20 mm over the pads' tops.
HANDLE_X = 0.034
HANDLE_Y0, HANDLE_Y1 = -0.070, 0.060
HANDLE_Z0, HANDLE_Z1 = 0.014, 0.048
HANDSET_MASS = 0.17
HANDSET_FRICTION = (0.45, 0.40)  # ABS (est)
HANDSET_CHARCOAL = (0.10, 0.10, 0.12)
HANDSET_LEN = (EAR_PAD[0][1] + EAR_PAD[1][1] / 2) - (MOUTH_PAD[0][1] - MOUTH_PAD[1][1] / 2)  # 0.197

# -- where the handset starts: face down on its pads, long axis along x, earpiece end toward +x --------------------
HANDSET_X, HANDSET_Y = 0.32, -0.22  # its frame's origin on the desk
HANDSET_YAW = -math.pi / 2  # its frame's +y (the earpiece end) points along the world's +x
GRIP_Y = -0.010  # where it is held, in its frame: about its centre of mass (-11 mm), clear of both pads
GRIP_Z = 0.034  # the TCP's height over the pad bottoms: pads (22 mm long) from 21 to 43 mm up the 14-48 mm handle
GRIP_W = HANDLE_X  # the fingers close across the handle's 34 mm

# -- the surroundings (fixed) -----------------------------------------------------------------------------------------
# A desktop monitor behind the phone (flat panel on a foot and neck) and a pen pot beside it.
MON_X = 0.56
MON_FOOT = (0.16, 0.24, 0.020)  # x, y, z
MON_NECK = (0.03, 0.06, 0.130)
MON_PANEL = (0.025, 0.52, 0.31)
MON_PANEL_Z0 = 0.12  # the panel's lower edge over the desk
MON_BLACK = (0.07, 0.07, 0.08)
POT_XY = (0.36, 0.175)
POT_R, POT_H = 0.040, 0.100
POT_BLUE = (0.20, 0.30, 0.55)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SLACK = 0.0005  # a pad corner counts as inside a cup within 0.5 mm of its wall
SEAT_LOW, SEAT_HIGH = -0.002, 0.003  # every pad corner within 2 mm below to 3 mm above the cup's floor
CUP_HIGH = 0.006  # 'in its cup' for the part credit: corners within 2 mm below to 6 mm above the floor
SEAT_TILT_DEG = 3.0  # the handset's turn out of level against the phone
SEAT_YAW_DEG = 6.0  # its heading against the phone's (plan view)
HOOK_DOWN = 0.002  # the plunger at least this far down: the line is closed
REST_HIGH = CUP_DEPTH + 0.004  # 'resting on the phone': pad corners no more than this over the cup floor (the top is 10 mm)
OFF_DROP = 0.010  # a part's lowest point this far below the table top: it fell off the desk
PHONE_MOVE = 0.008  # the phone's origin this far from its mark at any moment: shoved, 0
PHONE_TILT_DEG = 3.0  # or tilted this much
LET_GO = 0.030  # both grippers' finger zones this far from the handset and the phone
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # each part moves under 5 mm over the window (visible motion; the GPU demo ran at 0.0 mm)
STILL_TURN_DEG = 5.0  # and turns under 5 degrees
EPS = 1e-9
CONTROL_HZ = 20.0
START_TOL = 0.003  # the parts within 3 mm of their marks: the starting layout
RESET_JUMP = 0.08  # a part that lands on its mark having moved this much in one control step: a new episode began

PAD_TIP = 0.013  # the pad tips are 13 mm below the TCP

# -- spawn poses (the frames above) ----------------------------------------------------------------------------------
PHONE_ORIGIN = (PHONE_X, PHONE_Y, TOP)
HANDSET_ORIGIN = (HANDSET_X, HANDSET_Y, TOP)
SEATED_Z = TOP + FLOOR_Z  # the handset's frame when it is seated


def pad_corners(name):
    """The four bottom corners of a pad (counter-clockwise from +x +y) in the handset's frame, z 0."""
    (cx, cy), (sx, sy), _h = PADS[name]
    return tuple((cx + a * sx / 2, cy + b * sy / 2, 0.0) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)))


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return ((c, -s), (s, c))
