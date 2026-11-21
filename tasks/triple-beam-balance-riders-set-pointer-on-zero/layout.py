"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

An OHAUS 750-S0 class triple beam balance (README, Sources) stands on the table, fixed on its base: a pillar with the
knife edge, a beam that turns on it, the pan on a stem from the beam's left end, three parallel beams to the right
(front: the smooth 0 to 10 g beam with its poise; middle: the 0 to 500 g beam, 100 g notches, with its 100 g rider; rear:
the 0 to 100 g beam, 10 g notches, with its 10 g rider), a pointer at the right end against a fixed scale post. Sourced
figures are marked (src), estimates (est).

Beam angle THETA is the joint value of the beam about +x: positive lifts the pan side (+y) and drops the pointer end.
A heavy specimen on the pan therefore gives a NEGATIVE angle (pan down, pointer end high).
"""

import math

TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- placement ----------------------------------------------------------------------------------------------------------
X0 = 0.270  # the middle beam's x
YP = -0.200  # the knife edge's y: pan side to its left (+y), beams to its right (-y)
BEAM_DX = {"poise": -0.036, "b500": 0.0, "b100": 0.036}  # the three beams: front, middle, rear (src: Fisher listing)
HZ = 0.100  # the knife edge / beam axis height (est)

# -- the instrument ----------------------------------------------------------------------------------------------------
LA = 0.120  # knife edge to the pan's axis (est; sets the lever ratio below)
BEAM_LEN = 0.270  # the beams' right end, from the knife edge
BAR_W, BAR_H = 0.008, 0.010  # a beam's section
PAN_R = 0.076  # 6 in (15.2 cm) fixed flat plate (src)
PAN_T = 0.004
PAN_Z = 0.145  # the pan plate's top (est)
BASE_Y0, BASE_Y1 = 0.215, -0.300  # the base plate's ends from the knife edge (y offset, positive left)
BASE_W, BASE_H = 0.120, 0.020
PILLAR = (0.030, 0.020, 0.092)  # x, y, top z
SCALE_D = 0.285  # the scale post's distance right of the knife edge
POINTER_D = 0.265
POINTER_DROP = 0.040  # the pointer's tip under the beam axis (zero mark height = HZ - POINTER_DROP)
ZERO_Z = HZ - POINTER_DROP
STOP_DEG = 8.0  # the beam's stops: +-8 degrees
MARK_DEG = 1.0  # one scale mark per degree
# The weights. Reading = (rider mass x its distance from the knife edge) / LA, so the real ratio of rider mass to notch
# spacing is kept: 100 g per 0.040 m for the 0.300 kg rider, 10 g per 0.015 m for the 0.080 kg rider, 10 g over 0.100 m
# for the 0.012 kg poise (est from the lever ratio; the real rider masses are on the OHAUS sheet).
ZERO_D = 0.020  # every rider's zero notch, from the knife edge (to the right)
R100_STEP, R100_N, R100_MASS = 0.040, 5, 0.300  # notches 0..5 -> 0..500 g
R10_STEP, R10_N, R10_MASS = 0.015, 10, 0.080  # notches 0..10 -> 0..100 g
POISE_SPAN, POISE_MASS = 0.100, 0.012  # 0..10 g
BEAM_MASS = 0.50  # the beam, pan, stem, pointer and tare weight together (est)
CW_MASS_HINT = 0.0  # the tare weight is sized in parts.py so the empty beam hangs at zero
RIDER_W = 0.026  # across the beam (x)
RIDER_L = 0.020  # along the beam (y), the notched riders
POISE_L = 0.016
RIDER_TOP = 0.034  # the rider's top over the beam axis
# Sensitivity and damping: a self-righting spring stands for the centre of gravity under the knife edge, joint damping
# for the magnetic damper. K_BEAM is the NET self-righting with the rock on the pan (0.5 g at the pan per degree). The
# drawn parts (pan plate, riders, poise, rock) sit above the knife edge and tip the beam over like an inverted pendulum
# (GPU run 1, 2026-11-21: the empty beam ran off zero at 1.16 per second, net -0.062 N m/rad, exactly as their centres
# of mass predict), so the joint's spring is K_BEAM plus that gravity term (parts.gravity_stiffness, about 0.235 N m/rad).
SENS_G_PER_DEG = 0.5
K_BEAM = SENS_G_PER_DEG * 1e-3 * G * LA / math.radians(1.0)  # N m/rad, net, rock on the pan
C_BEAM = 0.045  # N m s/rad: slightly overdamped, about 3 s to settle
# The notch detents (task.py _detents): within the capture distance the rider is pulled toward the notch by a stiff
# spring whose pull is capped at RIDER_PULL (about 2 N to push it out), so it sits in the notch within a few hundredths
# of a mm. A soft spring (run 1: 200 N/m) left the 100 g rider 3 mm short on the tilted beam (7.5 g at the pan) and,
# with the rider's own weight sliding it on a tilted beam, would make the level beam unstable again.
RIDER_FRICTION = 0.05  # N
RIDER_DETENT_K = {"r100": 2000.0, "r10": 2000.0}  # N/m near the notch
RIDER_DAMPING = {"r100": 20.0, "r10": 10.0}  # N s/m (about 0.4 of critical on the stiff spring)
RIDER_PULL = 2.0  # N: the capped pull, so about 2 N to push a rider out of its notch
RIDER_CAPTURE = {"r100": 0.010, "r10": 0.005}
POISE_FRICTION = 0.30  # N (est): it stays where it is pushed
POISE_TOP = 0.034  # the poise's top over the beam axis (as tall as the riders: the pad keeps it with the beam on either stop)
STEEL = (0.74, 0.75, 0.77)
BLACK = (0.10, 0.10, 0.11)
BLUE = (0.18, 0.32, 0.62)
RED = (0.80, 0.12, 0.10)
CREAM = (0.90, 0.88, 0.80)
GREY = (0.52, 0.52, 0.54)
GOLD = (0.75, 0.62, 0.20)
GRANITE = (0.48, 0.46, 0.47)

# -- the specimen ------------------------------------------------------------------------------------------------------
SPEC_MASS = 0.2364  # granite specimen, weighed once on a reference scale
SPEC_G = 236.4
# start: on the table in front of the balance
SPEC_X, SPEC_Y = 0.130, -0.170
SPEC_BODY = (0.062, 0.048, 0.026)  # x, y, z of the main body
SPEC_CAP = (0.040, 0.030, 0.008)
SPEC_H = SPEC_BODY[2] + SPEC_CAP[2]

# -- the answer and the scoring windows --------------------------------------------------------------------------------
K100_ANS, K10_ANS = 2, 3  # 200 g and 30 g
POISE_ANS = (SPEC_G - 100.0 * K100_ANS - 10.0 * K10_ANS) / 10.0 * POISE_SPAN  # 6.4 g -> 0.064 m
NOTCH_TOL = 0.004  # a rider within 4 mm of its notch counts as in it
READ_TOL_G = 0.5  # total within +-0.5 g
READ_TOL_G_MID = 2.0
ZERO_DEG = 1.0  # pointer within +-1 degree of the zero mark
MID_DEG = 3.0
REST_DEG = 0.5  # over the last 1 s the beam turned by under 0.5 degree
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # the specimen moved under 5 mm over the last 1 s
LET_GO = 0.030
FALL_DROP = 0.020  # the specimen's lowest point 20 mm under the table top: fell
ON_PAN_R = PAN_R - 0.010  # the specimen's centre within this of the pan's axis (in the beam frame)
RESET_JUMP = 0.10
EPS = 1e-9
