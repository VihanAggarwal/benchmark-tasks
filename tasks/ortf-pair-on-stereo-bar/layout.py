"""Every dimension of the task, in one place (metres, kilograms, degrees where named, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A location-recording set-up for a choir: a grey stereo-bar plate on the midline, a choir riser beyond it with a white
mark at its centre, and two small pencil-condenser microphones lying in their case pads, one in front of each arm.
ORTF (Office de Radiodiffusion-Television Francaise) is two cardioid microphones with their capsules 170 mm apart and
their axes splayed outward at an included angle of 110 degrees (55 degrees either side of the pair's centre line),
the pair centred on the source and its centre line aimed at the source's centre. Spacing and angle are full scale.

A mic is one loose rigid body (a mechkit compound): a matte-black body and, at its front, a silver grille end. Its base
frame has its origin at the middle of its bottom face, local +x along the mic toward the grille (the capsule's axis,
'u'), local +z up. The capsule point is the grille's centre: 7.5 mm behind the front face, at mid-height.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the microphones (two alike, loose) ---------------------------------------------------------------------------
MIC_LEN, MIC_W, MIC_H = 0.070, 0.024, 0.024  # 70 mm long, 24 x 24 mm across
GRILLE_LEN = 0.015  # the front 15 mm is the silver grille
MIC_MASS = 0.080
BODY_CENTER = (-(MIC_LEN / 2) + (MIC_LEN - GRILLE_LEN) / 2, 0.0, MIC_H / 2)  # (-0.0075, 0, 0.012)
BODY_SIZE = (MIC_LEN - GRILLE_LEN, MIC_W, MIC_H)  # 55 x 24 x 24 mm
GRILLE_CENTER = (MIC_LEN / 2 - GRILLE_LEN / 2, 0.0, MIC_H / 2)  # (0.0275, 0, 0.012)
GRILLE_SIZE = (GRILLE_LEN, MIC_W, MIC_H)
CAPSULE_LOCAL = GRILLE_CENTER  # the capsule point: the grille's centre, 27.5 mm in front of the body's centre
CENTER_LOCAL = (0.0, 0.0, MIC_H / 2)  # the mic's centre (and its centre of mass)
TAIL_BACK = 0.0625  # the tail point t = c - 62.5 mm u (the body's rear face centre)
MIC_HALF = (MIC_LEN / 2, MIC_W / 2, MIC_H / 2)
MICS = ("mic_a", "mic_b")  # mic A starts in front of the left arm (+y), mic B in front of the right arm

# -- the case pads the mics start in (fixed grey foam, 1 mm thick, turned with their mic) -------------------------
PAD_SIZE = (0.100, 0.050, 0.001)
PAD_T = PAD_SIZE[2]
MIC_START = {  # name -> (x, y of the mic's centre, capsule yaw in degrees: the way its grille end points, CCW from +x)
    "mic_a": (0.250, 0.320, -20.0),
    "mic_b": (0.250, -0.320, 20.0),
}

# -- the stereo-bar plate (fixed, grey aluminium) ------------------------------------------------------------------
PLATE_X0, PLATE_X1 = 0.260, 0.340  # 80 mm along x
PLATE_Y0, PLATE_Y1 = -0.140, 0.140  # 280 mm along y
PLATE_H = 0.008
PLATE_TOP_W = TOP + PLATE_H

# -- the choir riser (fixed, dark-blue painted wood) with the centre mark and six singer figures -------------------
RISER_X0, RISER_X1 = 0.470, 0.500  # 30 mm along x
RISER_Y0, RISER_Y1 = -0.150, 0.150  # 300 mm along y
RISER_H = 0.040
MARK = (0.485, 0.000)  # the white mark at the choir's centre, on the riser top
MARK_RADIUS = 0.006  # 12 mm across
MARK_T = 0.0005
FIGURE_RADIUS = 0.008  # singer figures: upright painted wooden pegs, 16 mm across
FIGURE_H = 0.030
FIGURE_Y = (-0.120, -0.080, -0.040, 0.040, 0.080, 0.120)
SOURCE_AXIS_Y = MARK[1]  # the source axis: the line through the mark along x (y = 0)

# -- the rule (scoring.py) -----------------------------------------------------------------------------------------
ANGLE_DEG, ANGLE_TOL_DEG = 110.0, 8.0  # the included angle between the two capsule axes, seen from above
SPACING, SPACING_TOL = 0.170, 0.010  # the capsule points' separation
FACING_DEG = 30.0  # A: the pair's bisector within 30 degrees of the direction from the capsules' midpoint to the mark
AIM_DEG = 5.0  # C: ... within 5 degrees
CENTRE_TOL = 0.010  # C: the capsules' midpoint within 10 mm of the source axis (y = 0)
FLAT_DEG = 10.0  # gate: each mic's long axis within 10 degrees of horizontal
ON_PLATE_TOL = 0.005  # gate: each mic's lowest point within 5 mm of the plate top (not on the other mic, not the table)
HANDS_OFF = 0.010  # gate: every gripper's finger zone at least 10 mm from each mic (let go)
STILL_WINDOW_S = 0.5  # gate: still: in each of the last two 0.5 s windows the mic's centre moved (path) under ...
STILL_PATH = 0.0015  # ... 1.5 mm, ...
STILL_YAW_DEG = 2.0  # ... and its heading changed less than 2 degrees over the last 1 s
STILL_WINDOWS = 2
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a mic landing on a starting mark having moved this much further in one control step than its
#                    last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on a starting mark': within this of it (covers a seeded reset's +-3 cm jitter)
START_YAW_TOL_DEG = 10.0
TIERS = (0.33, 0.67, 1.0)

# -- the grippers (the let-go rule; rigstate reports the TCP, the point between the pad tips and the opening) -----
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- the demo's end state (task.py aims at it; tests check it scores 1.0) ------------------------------------------
TARGET_CAPSULE = {"mic_a": (0.320, 0.085), "mic_b": (0.320, -0.085)}
TARGET_YAW_DEG = {"mic_a": 55.0, "mic_b": -55.0}

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
MIC_BLACK = (0.05, 0.05, 0.05)  # matte-black anodised body
GRILLE_SILVER = (0.82, 0.83, 0.85)  # silver steel-mesh grille
FOAM_GREY = (0.42, 0.42, 0.44)  # case pads
PLATE_GREY = (0.60, 0.61, 0.63)  # bar plate
RISER_BLUE = (0.08, 0.12, 0.35)  # riser
MARK_WHITE = (0.97, 0.97, 0.97)
FIGURE_MAROON = (0.55, 0.12, 0.16)  # singer figures


def unit(deg: float) -> tuple[float, float]:
    return math.cos(math.radians(deg)), math.sin(math.radians(deg))


def centre_for(capsule_xy, yaw_deg: float) -> tuple[float, float]:
    """The mic centre (x, y) that puts its capsule point at ``capsule_xy`` with its grille end pointing ``yaw_deg``."""
    ux, uy = unit(yaw_deg)
    d = CAPSULE_LOCAL[0] - CENTER_LOCAL[0]
    return capsule_xy[0] - d * ux, capsule_xy[1] - d * uy


TARGET_CENTRE = {n: centre_for(TARGET_CAPSULE[n], TARGET_YAW_DEG[n]) for n in MICS}  # (0.3042, +-0.0625)
