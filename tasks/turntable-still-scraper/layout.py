"""Every dimension and threshold of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say TOP.

The cake turntable is one fixed mechkit assembly: a low base fixed to the table and, on a revolute joint about the
vertical axis through AXIS, the plate with the dummy cake, four pegs and an index mark on it (the link "plate"; its
joint angle, positive anticlockwise seen from above, is how far the turntable has turned). The cake smoother is a
loose mechkit compound (a blade with a handle along its top edge), hanging by its handle in a fixed slotted stand.
"""

import math

TOP = 0.7475  # station table top

# -- the turntable (fixed assembly; base frame origin on the table top at the axis) -----------------------------
AXIS = (0.270, -0.050)  # its vertical axis (world x, y)
BASE_R, BASE_H = 0.050, 0.005  # the cast base: a disc 100 mm across, 5 mm tall
PLATE_R, PLATE_T = 0.170, 0.010  # the turning plate, 340 mm across, 10 mm thick, its underside on the base
PLATE_TOP = BASE_H + PLATE_T  # 0.015
CAKE_R, CAKE_H = 0.060, 0.050  # the dummy iced cake on the plate's centre: 120 mm across, 50 mm tall
CAKE_TOP = PLATE_TOP + CAKE_H  # 0.065
PEG_R, PEG_H = 0.010, 0.030  # four pegs to turn it by, 20 mm across, 30 mm tall
PEG_TOP = PLATE_TOP + PEG_H  # 0.045
PEG_CIRCLE = 0.155  # their centres' distance from the axis (their outer edges 5 mm in from the rim)
PEG_ANGLES_DEG = (30.0, 120.0, 210.0, 300.0)  # where they stand at the start, anticlockwise from forward (+x)
MARK_SIZE = (0.020, 0.008, 0.001)  # a red index mark on the plate, along its radius at 0 deg (forward) ...
MARK_AT = 0.155  # ... its centre this far from the axis
PLATE_MASS = 0.93  # the plate, the cake and the pegs together (HDPE plate 862 g, acetal pegs 53 g, foam dummy 17 g)
PLATE_DAMPING = 0.10  # N m s/rad: the bearing's drag (no spring: it stays where it is left). With the link's
#                       ~0.0093 kg m^2 it coasts with a 0.09 s time constant: let go at 1 rad/s it stops in 5 deg
PLATE_LOWER, PLATE_UPPER = -0.20, 6.10  # rad: the joint's travel, stops at 11.5 deg clockwise and 349.5 deg
#                                         anticlockwise (inside +-2 pi, where the simulator's revolute limits hold)

# -- the cake smoother (loose compound; its base frame origin at the middle of the blade's bottom edge) ----------
BLADE = (0.100, 0.006, 0.070)  # clear acrylic: 100 mm along its length (local x), 6 mm thick, 70 mm tall
HANDLE = (0.070, 0.032, 0.030)  # hollow black polypropylene (2 mm walls), centred on the blade's top edge
SMOOTHER_H = BLADE[2] + HANDLE[2]  # 0.100
SMOOTHER_MASS = 0.067  # blade 50 g (acrylic, 1.18 g/cm3) + handle 17 g
# The stand: two fixed grey blocks along x leaving an 18 mm slot. The blade hangs in the slot (6 mm clear each side,
# its bottom 2 mm above the table) and the handle rests across the blocks' tops (7 mm of it on each, more than the
# blade's 6 mm of play, so it stays on both whichever way it is pushed).
STAND_XY = (0.270, 0.215)  # the slot's centre; the smoother's start (length along x)
STAND_BLOCK = (0.120, 0.012, 0.072)
STAND_SLOT = 0.018
HANG = STAND_BLOCK[2] - BLADE[2]  # 0.002: the blade's bottom above the table while it hangs in the stand
SMOOTHER_START = (STAND_XY[0], STAND_XY[1], TOP + HANG)  # its origin at the start

# -- where the demo sets the smoother (the score does not depend on where it is set) ------------------------------
SET_ANGLE_DEG = 60.0  # the contact point: the cake's front left (the left arm's links then stay 4 cm or
#                       more from the right arm's while it turns the plate)
SET_GAP = 0.0008  # blade face to the cake's side, as commanded (the held smoother's ~0.2 mm of sag outward adds
#                   to it: J K^-1 J^T of the arm's joint gains, the 67 g payload not compensated)
SET_LIFT = 0.0035  # blade bottom above the plate, as commanded (its ~1.8 mm of sag downward takes from it)
APPROACH_OUT = 0.020  # the blade comes in to the side from this much further out (and leaves the same way)
GRIP_UP = 0.009  # the finger tips this far above the handle's underside: 21 mm of the 30 mm handle in the pads
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_TCP = BLADE[2] + GRIP_UP + PAD_TIPS  # 0.092: the TCP above the smoother's origin when it is held

# -- scoring (scoring.py) --------------------------------------------------------------------------------------
N_SECTORS = 360  # the cake's side in one-degree sectors (the cake's own frame: they turn with it)
ARC_DEG = 3.0  # the credited contact arc: the side within this either way of the contact point (3 mm of side)
FULL = 340  # 1.0 (with the finish): sectors that passed the blade (the turntable's stop allows 349.5 deg of turn,
#             so a full 360 cannot pass; one uninterrupted turn of D deg passes about D + 6)
THREE_QUARTER = 270  # 0.75
HALF = 180  # 0.5
SET_HOLD_S = 1.0  # 0.25: set this long without a break
HOLD_OPENING = 0.5  # 'held': a gripper at most this open (0 closed .. 1 open: 47 mm) ...
HOLD_TOL = 0.010  # ... with its finger-tip point within this of the handle (the handle's box, in its own frame)
TILT_MAX_DEG = 5.0  # 'set': the smoother upright within this
GAP_MIN, GAP_MAX = -0.001, 0.003  # ... the blade this far from the cake's side where they are nearest (its real box)
FACE_MAX_DEG = 45.0  # ... its length within this of the side's tangent there (flat on it, or angled, edge on)
BOTTOM_MIN, BOTTOM_MAX = -0.001, 0.005  # ... the blade's lowest point this far above the plate's top
STILL_SPEED = 0.010  # ... the blade's centre moving slower than this (m/s) ...
STILL_WINDOW = 0.010  # ... and within this of where it was when the smoother was put in place
STILL_TURN_DPS = 3.0  # ... its heading turning slower than this (deg/s) ...
HEADING_WINDOW_DEG = 4.0  # ... and within this of its heading when it was put in place. Once out of either
#                           window it is not set again until it has been taken away:
AWAY = 0.020  # its centre this far from where it was put, or its bottom this far above the plate, or let go
DRIVE_GAP = 0.005  # coverage counts only while the OTHER gripper's finger zone is within this of a peg (turning it)
TURN_NOISE_DEG = 0.01  # a control step's turn smaller than this is no turn (solver noise)
TURN_GLITCH_DEG = 20.0  # ... and a jump bigger than this in one step is not counted (a bad reading)
FINGER_CAKE_MIN = 0.005  # 0: a gripper's finger zone came this close to the cake (side or top)
TOP_TOUCH = 0.003  # 0: the smoother within this of the cake's top surface, over it (the top's disc ...
TOP_INSET = 0.003  # ... shrunk by this, so a blade pressed 1 mm into the side is not 'on the top')
OFF_TABLE = 0.020  # 0: the smoother's lowest point this far or more below the table top (it fell off)
FINISH_CAKE = 0.010  # finish: the smoother at least this far from the cake ...
FINISH_PEG = 0.020  # ... every gripper's finger zone at least this far from every peg ...
STILL_TURN = 0.05  # ... and the turntable turning slower than this (rad/s) ...
FINISH_HOLD_S = 1.0  # ... all held this long (the same as task.SUCCESS_HOLD_S)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
# The finger zone covers the lowest 38 mm of each finger (yam_arena's finger collision meshes: 6.4 mm thick and
# 27 mm wide at the tips, 10 mm thick and 40 mm wide 19-38 mm up), as a round zone about the gripper's axis (the
# scorer cannot see its heading): half the gap plus 10 mm, and half of 40 mm across.
FULL_GAP = 0.0939
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
FINGER_LEN = 0.038

CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
# Reset fallback (rigstate's and mechkit's reset hooks normally come first): the smoother jumps, in one control
# step, this much further than its velocity allows, back to its start in the stand, with the turntable at 0.
RESET_JUMP = 0.05
START_TOL = 0.010
START_TURN_TOL = math.radians(1.0)

# -- the offline stand-in for the turntable's joint (offline_model.py; the simulator never uses it) --------------
GRIP_CAPTURE = 0.010  # a gripper closing past a peg's width with its finger-tip point this close to the peg ...
GRIP_SLIP = 0.015  # ... carries the plate round with it until it opens or drifts this far off the peg's circle

# Colours (sRGB).
GREY = (0.55, 0.56, 0.58)  # the base (cast aluminium)
WHITE = (0.93, 0.93, 0.92)  # the plate
CREAM = (0.97, 0.92, 0.80)  # the iced cake
BLACK = (0.08, 0.08, 0.09)  # the pegs, the handle
RED = (0.85, 0.12, 0.10)  # the index mark
CLEAR = (0.72, 0.84, 0.90)  # the acrylic blade
STAND_GREY = (0.40, 0.41, 0.44)
