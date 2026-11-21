"""Every number of the scene, in metres, kg, N, N m and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up; the table top (the bench top) is at z TOP.
"Table frame" in comments is the world with z measured from the table top. The router's frame R has its origin on
the spindle axis at the bench top, axes the world's. The wrench's frame W has its origin at the centre of its open
jaw (where the nut's axis sits when the jaw is on), x along the handle, z up; the wrench lies flat.

Figures marked est in task.json are design values the simulation is built to; the setup says how the real parts are
matched to them.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # the station table top (world z)
CONTROL_HZ = 20.0
HOLD_STEPS = 20  # 1 s at 20 Hz: the end state must hold this long

# -- colours (sRGB 0..1) ------------------------------------------------------------------------------------------
BOSCH_BLUE = (0.05, 0.27, 0.55)  # the Colt motor housing
BLACK = (0.08, 0.08, 0.09)  # housing bands, cord, plug, lock button rubber
STEEL = (0.72, 0.73, 0.75)  # collet nut, spindle, wrench
BRASS = (0.78, 0.62, 0.28)  # the stand-in bit's painted cutter (a ground-off bit, painted so it reads on camera)
BENCH_WOOD = (0.62, 0.46, 0.30)  # bit block, wrench rest, cradle ring
WELL_DARK = (0.18, 0.15, 0.12)  # the bench well's rim
STRIP_WHITE = (0.93, 0.93, 0.91)  # power strip
RED = (0.80, 0.10, 0.10)  # the strip's switch

# -- the router motor (fixed): Bosch PR20EVS Colt, out of its base, collet up in a bench well -------------------------
RX, RY = 0.230, 0.000  # the spindle axis (table frame)
# The router's frame R may be turned ROUTER_YAW about z. As built, R's -x (where the lock button is) faces the robot:
# the right arm presses the button straight forward (+x, near its own reach line) and the left arm works the wrench
# on the router's far left side, so the two forearms never cross.
ROUTER_YAW = 0.0
HOUSING_R = 0.0375  # 75 mm across (Bosch: 2.9 in grip)
HOUSING_TOP = 0.090  # the housing's top shoulder above the bench (the motor's lower part is in the well)
BAND_Z = (0.066, 0.084)  # the lock band: the housing cut away round the lock pin's channel
CHANNEL_HALF_Y = 0.008  # the pin's channel through the band, toward -x
CHANNEL_X_IN = -0.020  # the channel runs from the housing's -x face in to here (R frame)
SHAFT_R = 0.008
SHAFT_Z = (BAND_Z[0], 0.100)  # the spindle shaft, up through the housing to the nut
NUT_Z = (0.100, 0.115)  # the 1/4 in collet nut, 17 mm across flats (as modelled), 15 mm tall
NUT_FLATS = 0.017
NUT_BORE_R = 0.00425  # the hex ring's inscribed bore (the collet's mouth): the shank (3.175 mm) has 1 mm play
# The lock disc on the spindle inside the band: 6 teeth with a gap at 180 + k 60 deg (the pin enters at 180).
TEETH = 6
TOOTH_Z = (0.0705, 0.0795)
TOOTH_R = (0.012, 0.030)
TOOTH_W = 0.016  # tangential: the gap at the rim is 31.4 - 16 = 15 mm, the pin 6 mm
SPINDLE_DAMPING = 0.002  # N m s/rad: the armature on its bearings, est
SPINDLE_MASS = 0.08
NUT_MASS = 0.02
# The collet nut's turn on the spindle: joint friction. The real nut is reset to the same 0.5 N m before every trial
# (torque screwdriver and 17 mm socket); the stand-in bit carries no cutting load. Loosening is anticlockwise seen from
# above: + about z.
NUT_FRICTION = 0.5
NUT_UPPER = math.radians(300.0)  # inside mechkit's 2 pi
BIT_FREE_AT = 0.7  # rad (40 deg) of the nut on the spindle: the collet lets go of the shank (the bond is released)

# -- the spindle-lock button (fixed housing guide, a sprung slider) -----------------------------------------------------
BUTTON_Z = 0.075  # its centre above the bench (on R's -x side, world +y, under the nut)
BUTTON_SIZE = (0.008, 0.014, 0.012)  # x (thick), y, z: the rubber cap
BUTTON_FACE_X = -0.0445  # its outer face at rest (R frame): 7 mm proud of the housing
PIN_SIZE = (0.0045, 0.006, 0.006)  # x, y, z: from the cap's inner face in
PIN_TIP_X = -0.032  # rest (R frame): 2 mm short of the teeth's rim
BUTTON_TRAVEL = 0.005  # to its stop: the pin 3 mm into a gap
BUTTON_SPRING = 1200.0  # N/m: 6 N at the stop, est
BUTTON_DAMPING = 2.0
LOCKED_AT = 0.0035  # the pin 1.5 mm or more into the gap: the spindle cannot turn



def to_router(p) -> np.ndarray:
    """A world point (table-frame xy, any z) in the router's frame R (xy only)."""
    c, s = math.cos(-ROUTER_YAW), math.sin(-ROUTER_YAW)
    d = np.asarray(p, dtype=float)[:2] - np.array([RX, RY])
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def from_router(q) -> np.ndarray:
    """A point of R's xy plane in world xy."""
    c, s = math.cos(ROUTER_YAW), math.sin(ROUTER_YAW)
    q = np.asarray(q, dtype=float)
    return np.array([RX + c * q[0] - s * q[1], RY + s * q[0] + c * q[1]])


# -- the stand-in bit (loose): a blunt 1/2 in straight bit on a 1/4 in shank, flutes ground off --------------------------
SHANK_R = 0.003175
SHANK_Z = (0.101, 0.121)  # seated: 14 mm in the collet's mouth, 6 mm above the nut
CUTTER_R = 0.00635
CUTTER_Z = (0.121, 0.141)
BIT_MASS = 0.030
BIT_FRICTION = (0.6, 0.5)

# -- the wrench (loose): the 17 mm collet wrench supplied with the router, open jaw, stamped steel --------------------
WRENCH_T = 0.004
WRENCH_MASS = 0.055
JAW_GAP = 0.0173  # 0.3 mm play on the 17 mm flats
PRONG_W = 0.006
PRONG_X = (-0.012, 0.0103)  # W frame: prongs from the jaw's mouth back to the throat
HEAD_X = (0.0103, 0.024)
HEAD_HALF_Y = JAW_GAP / 2 + PRONG_W
HANDLE_X = (0.024, 0.140)
HANDLE_W = 0.018
GRIP_X = 0.090  # the left hand pinches the handle here (across its 18 mm width), 26 mm from the wrench's centre of
# mass (65 mm from the jaw): pinched 125 mm out on the GPU (20 Nov) it pitched jaw-down about the pads in the hand
WRENCH_FRICTION = (0.6, 0.5)
# Its rest: flat on a wooden rest block, jaw toward -x; the handle overhangs the block's +x end by 5 cm.
WRENCH0 = np.array([0.180, 0.260])  # jaw centre (table frame xy), yaw 0 (handle toward +x)
REST_X = (WRENCH0[0] - 0.014, WRENCH0[0] + 0.090)
REST_H = 0.025
WRENCH_Z_REST = REST_H + WRENCH_T / 2  # its mid-plane on the block
WRENCH_Z_NUT = (NUT_Z[0] + NUT_Z[1]) / 2  # its mid-plane on the nut

# -- the bit block (fixed): a wooden block with three 10 mm holes -----------------------------------------------------
BLOCK_C = np.array([0.230, 0.335])
BLOCK_SIZE = (0.060, 0.030, 0.020)
HOLE_D = 0.010  # 1.8 mm of radial clearance round the 1/4 in shank (was 9 mm, 1.3: widened 21 Nov before the bit's
# first free lift on the GPU; the GPU hand sat within 1 mm at this pose); the 12.7 mm cutter still rests on the top
HOLES = [np.array([BLOCK_C[0] + dx, BLOCK_C[1]]) for dx in (-0.018, 0.0, 0.018)]
HOLE_USED = 1  # the demo's hole: the middle one

# -- the power strip (fixed), standing on its back edge, outlets facing the robot (-x); the plug on a slider ----------
STRIP_X = (0.180, 0.215)  # its outlet face at x 0.180
STRIP_Y = (-0.290, -0.090)
STRIP_H = 0.045
PLUG_Y = -0.150
PLUG_SIZE = (0.022, 0.025, 0.030)  # x, y, z: the plug body, in front of the face
PLUG_Z = 0.020  # body centre above the bench
PLUG_FRICTION = 3.0  # N: the receptacle contacts' hold, est (measured on the real strip with a spring scale). Was 6 N:
# on the GPU (20 Nov) a 6 N plug did not move under the pinched hand's pull (about 8 N, the TCP 15 mm behind its
# target), so the hold is set to what the arm's pull (about 8 N across its line) beats with margin
PLUG_UPPER = 0.030  # it can be drawn 30 mm out
BLADE_LEN = 0.016  # the blades' length: out 16 mm or more, the plug is clear of the contacts
UNPLUGGED_AT = BLADE_LEN
CORD_R = 0.0035

# -- arms ---------------------------------------------------------------------------------------------------------
FULL_GAP = 0.0939
PAD_TIP = 0.013
PAD_THICK = 0.008
OPEN_GAP = 0.060  # fingers open for every approach
PLUG_GAP = 0.017  # pinch the 25 mm plug body (8 mm under its width)
HANDLE_GAP = 0.012  # pinch the 18 mm handle
BIT_GAP = 0.006  # pinch the 12.7 mm cutter

# -- scoring ------------------------------------------------------------------------------------------------------
NEAR_BIT = 0.010  # while plugged in: no finger this close to the bit
ENGAGE_R = 0.006  # the wrench's jaw centre within this of the nut's axis, at nut height: the wrench is on the nut
ENGAGE_DZ = 0.010
OUT_OF_COLLET = 0.002  # the shank's bottom this far above the nut's top: the bit is out
IN_HOLE_XY = 0.003
IN_HOLE_DEPTH = 0.010  # the shank's bottom at least this far below the block's top
HOLE_TILT = math.radians(15.0)
WRENCH_DOWN_Z = 0.006  # the wrench's lowest point within this of the bench (or 4 mm over the rest block)
WRENCH_CLEAR_R = 0.050  # its jaw at least this far from the spindle axis
HANDS_AWAY = 0.050
STILL = 0.005  # mm-scale motion allowed over the last 1 s (m)
FELL_Z = -0.020  # a part this far under the bench top fell off

# -- demo geometry ------------------------------------------------------------------------------------------------
STROKE_FROM = math.radians(7.0)  # world heading of the handle when the jaw goes on (in front of the router, toward +x)
STROKE_TO = math.radians(76.0)  # ONE 69 deg stroke (1.20 rad), anticlockwise seen from above; joint 6 of the left arm
# is at its limit by 80 deg with the 90 mm grip. Why one long stroke (21 Nov, from the GPU traces of jobs 142744 and
# 145559): a 51 deg stroke from 25 deg turned the nut 0.644 rad on the spindle (0.89 commanded: 0.145 spindle play
# before the half-pressed lock pin caught a tooth, about 0.05 jaw seating, 2.7 deg of wrist trail); a second stroke
# never took hold because the wrench came off the nut hanging about 9 mm lower at the jaw in the pads (wrench origin z
# 0.845 against 0.854 on the nut, the hand's height unchanged), so the re-seated jaw passed UNDER the nut, round the spindle shaft. Every hand pose of a
# 7 to 76 deg stroke at the 90 mm grip was reached on the GPU in job 145559 (the second stroke went on at 7 deg, the
# first ended at 76 deg) with the lock hand undisturbed. Expected on the GPU: 1.20 - 0.145 - 0.10 = 0.96 rad, 0.26
# past the collet's 0.7 in a single engagement.
NUT_FLAT_PHASE = STROKE_FROM - ROUTER_YAW + math.radians(30.0)  # R frame: flat normals at 37 + k 60 deg, so the
# jaw fits a handle heading of STROKE_FROM (normals at heading +-90 deg). The nut's start angle is arbitrary on the
# real router: it is set at reset with the marks (reset_notes)
LOCK_PLAY = 0.145  # rad: the spindle's free turn before the lock pin (2.9 mm in on the GPU) catches a tooth
SEAT_TRAIL_LOSS = 0.10  # rad of each engagement's commanded turn the nut does not get: jaw seating (about 0.05)
# and the wrist trailing (2.7 deg), measured on the GPU (job 142744); the offline model charges it (offline_router.py)
GRIP_TIP_DZ = -0.0005  # the pad tips 0.5 mm under the wrench's mid-plane: on the rest 1.5 mm over the block's top
SLIDE = 0.025  # the jaw's slide on and off the nut along the handle
PAD_HALF_W = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's height, 15 mm at the tip
PRESS_EDGE = 0.0155  # the lock hand (yaw 0, rolled, fingers shut): the pads' +x edge this far ahead of the TCP at the
# cap; measured on the GPU (20 Nov, job 134455: the cap 2.9 mm in with the TCP at x 0.173)


def axis_xy() -> np.ndarray:
    return np.array([RX, RY])


def unit(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])
