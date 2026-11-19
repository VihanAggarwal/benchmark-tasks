"""Every number of the scene, in metres, kg, N and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up; the table top is at z TOP. Heights in the
comments are above the table top. The mannequin head faces the arms (toward -x), so the child's LEFT is world -y.

The headset's base frame H (its base body is the headband) has its origin on the band's centre line at the height
of the band's underside; its axes are the world's at rest. Its links: cup_p (the +y ear cup, the child's right),
cup_n (the -y ear cup, the child's left) and boom (the microphone boom, on cup_n). Every shape is drawn with every
joint at 0: cups at their sprung-in rest, boom stowed straight up.

Figures marked est in task.json are design values the simulation is built to; the setup's checks say how the real
parts are matched to them.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # the station table top (world z)
G = 9.81
CONTROL_HZ = 20.0
TABLE_X = (-0.19, 0.91)
TABLE_Y = (-0.65, 0.65)

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
FOAM = (0.93, 0.92, 0.89)  # white polystyrene head form
LIP = (0.80, 0.22, 0.25)  # the mouth line drawn on the form
EYE = (0.20, 0.20, 0.22)
BLACK = (0.07, 0.07, 0.08)  # headset plastic
GREY = (0.32, 0.33, 0.35)  # cushions (leatherette)
SILVER = (0.70, 0.71, 0.73)  # the stand

# -- the child-size head form (polystyrene, 52 cm round), fixed to the table on its flat neck base ---------------
HX, HY = 0.330, 0.0  # the head's vertical axis (world)
HEAD_ORIGIN = (HX, HY, TOP)
# stacked turned sections (z0, z1, radius): neck, lower face, cranium, three crown steps
HEAD_SECTIONS = [
    (0.000, 0.045, 0.045),
    (0.045, 0.085, 0.058),
    (0.085, 0.160, 0.070),
    (0.160, 0.185, 0.060),
    (0.185, 0.200, 0.042),
    (0.200, 0.205, 0.020),
]
CROWN_Z = 0.205
EAR_Y0, EAR_Y1 = 0.070, 0.080  # each ear: from the cranium's side out to y 80 mm (outer face)
EAR_X = 0.025  # along x (front to back)
EAR_Z = (0.0975, 0.1325)  # 35 mm tall
EAR_ZC = sum(EAR_Z) / 2  # 115 mm
MOUTH_Z = 0.070
MOUTH_HALF_W = 0.015
MOUTH_X = -0.058  # head frame: on the lower face's front
MOUTH_CORNER_LEFT = np.array([HX + MOUTH_X, HY - MOUTH_HALF_W, TOP + MOUTH_Z])  # the child's left corner (world)

# -- the headset (Jabra Evolve2 30 SE stereo class: 102 g without cable) ----------------------------------------
BAND_W = 0.024  # the band's width along x
BAND_T = 0.012
PIVOT_Y = 0.097  # each yoke's top, where it meets the band (H)
PIVOT_Z = -0.045
CUP_ZC = -0.090  # cup centre in H
CUP_R = 0.030  # shell radius (60 mm across)
SHELL_Y = (0.080, 0.102)  # shell, from the cushion's back out (|y| in H)
CUSH_R = 0.027
CUSH_Y = (0.068, 0.080)  # cushion: 12 mm thick; inner face at |y| 68 mm at rest
REST_INNER = CUSH_Y[0]
# The band's flex is modelled as each cup sliding straight out along y on a sprung slide (the cups stay upright,
# as the real cups do on their swivels): 167 N/m per side, est, so 2.0 N per side when seated on the ears.
CUP_SPRING = 167.0  # N / m toward the rest pose (each cup)
CUP_UPPER = 0.035  # m outward
CUP_DAMP = 2.0  # N s / m
BAND_MASS = 0.050
CUP_MASS = 0.022
BOOM_MASS = 0.008
HEADSET_MASS = BAND_MASS + 2 * CUP_MASS + BOOM_MASS  # 0.102 kg
HEADSET_FRICTION = (0.8, 0.7)  # leatherette and soft-touch plastic on polystyrene, est

# Boom on cup_n: a hinge about y through the cup's centre line, just outside the shell (y -106 mm).
BOOM_PIVOT = (0.0, -0.106, CUP_ZC)
BOOM_BAR_X = 0.006  # the arm stands 6 mm behind the hinge axis when stowed: gravity holds it on its stop
BOOM_Y = -0.108
# Stowed, the tip arm and the microphone lie over the stand's cradle (y within 45 mm of the centre line). With a 96 mm
# boom the microphone's underside sat exactly on the cradle top, so the fixed cradle blocked the swing forward (first
# GPU run, 2026-11-19: the push yawed the whole headset 11 degrees on the cradle instead). At 104 mm every part of
# the boom clears the cradle by 6.8 mm over the whole swing (tests: Geometry.test_boom_swing_clears_the_stand).
BOOM_LEN = 0.104  # hinge to the tip arm's centre line
TIP_Y1 = -0.020  # the tip arm runs in from the boom (y -108) to the microphone at y -20 mm (at rest)
BOOM_DOWN = -2.06  # rad: its lower stop, the microphone in front of the mouth's left corner
BOOM_DAMP = 0.003
MIC_NEAR = 0.050  # scorer: microphone within 50 mm of the mouth's corner
BOOM_DOWN_TOL = 0.25  # scorer: boom within 0.25 rad of its lower stop


Q_SEAT = EAR_Y1 - REST_INNER  # 12 mm: cushions on the ears' outer faces
Q_SPREAD = Q_SEAT + 0.012  # 24 mm: 12 mm clear of each ear
# The demo commands 30 mm: the sideways pull (about 5 N at 30 mm) bends the soft wrist about 1.8 mm per N, so the
# cup ends up roughly 21 mm out, still 9 mm clear of its ear.
SPREAD_CMD = 0.030
CLAMP_N = CUP_SPRING * Q_SEAT  # N at each cushion when seated

# -- the desk headphone stand (aluminium, single post) ----------------------------------------------------------
SX, SY = 0.150, 0.0
STAND_ORIGIN = (SX, SY, TOP)
STAND_TOP = 0.212  # the cradle's top: the band's underside rests here
SEATED_Z = CROWN_Z  # the band's underside on the crown

HEADSET_ORIGIN_START = (SX, SY, TOP + STAND_TOP)
HEADSET_ORIGIN_SEATED = np.array([HX, HY, TOP + SEATED_Z])

# -- the arms -------------------------------------------------------------------------------------------------------
FINGER_STIFF = 5000.0
OPEN_GAP = 0.086  # approach: 13 mm clear of the 60 mm shell on each side (the GPU wrists lag about 5 mm in x)
GRIP_GAP = 0.058  # 2 mm under the shell's 60 mm: 5 N a finger
GRIP_DZ = 0.0  # pad tips level with the cup's centre
PUSH_GAP = 0.010

# -- the scorer -----------------------------------------------------------------------------------------------------
EAR_NEAR = 0.015  # cushion face centre within 15 mm of its ear's centre across the ear (x and z)
EAR_OFF = 0.003  # and its face no more than 3 mm out from the ear's outer face (4 mm into it at most)
EAR_IN = 0.004
SEAT_TOL = 0.015  # the band's origin within 15 mm of the seated pose
TILT_MAX = 0.25  # rad
OFF_LIFT = 0.015  # off the stand: the band's underside 15 mm or more over the cradle's top,
OFF_SIDE = 0.030  # or its centre 30 mm or more outside the cradle's 30 x 90 mm top in plan
CRADLE_HALF = (0.015, 0.045)
BELOW_TOP = 0.010  # fell off the table (latch): any part 10 mm or more below the table top (contact allowance);
# with TABLE_X / TABLE_Y (any part past the table's edge). Set down on the table top is not a fall.
STILL_SPEED = 0.005
HANDS_OFF = 0.030
WINDOW = 20


def pinch_force(gap: float, width: float) -> float:
    return FINGER_STIFF * max(0.0, (width - gap) / 2)
