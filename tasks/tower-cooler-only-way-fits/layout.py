"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A bench model of a motherboard, laid flat at about half scale with its rear (I/O) edge away from the arms (+x): the
processor in the middle, two tall memory modules on its near side (-x), the VRM heatsink on its left (+y), a
graphics card standing in the top slot on its right (-y), the rear I/O shield along the far edge. Seen as it would
stand in a tower case, the case front is toward the arms and the rear exhaust beyond the I/O shield, so a cooler's
airflow should run along +x.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the board and what is fixed on it ---------------------------------------------------------------------------
BOARD_X0, BOARD_X1 = 0.225, 0.385  # 160 mm along x
BOARD_Y0, BOARD_Y1 = -0.085, 0.085  # 170 mm along y
BOARD_T = 0.004
BOARD_TOP = BOARD_T  # 4 mm

CPU_X, CPU_Y = 0.300, 0.000  # the processor's centre: the reference point of the whole layout
CPU_SIZE = (0.038, 0.038, 0.004)  # its lid, 38 x 38 mm, standing 4 mm proud of the board
CPU_HALF = CPU_SIZE[0] / 2
CPU_TOP = BOARD_TOP + CPU_SIZE[2]  # 8 mm

# The retention frame: four black bars round a 44 x 44 mm opening centred on the processor, 1 mm lower than the
# processor's lid (as a socket's load frame is), so the cooler's base sits on the lid alone and the frame marks
# where it goes.
FRAME_OPEN_HALF = 0.022  # the opening is 44 x 44 mm
FRAME_BAR_W = 0.003
FRAME_H = 0.003
FRAME_TOP = BOARD_TOP + FRAME_H  # 7 mm

# Two tall memory modules on the processor's near side, standing on the board along y.
MEM_SIZE = (0.004, 0.070, 0.030)  # 4 mm thick, 70 mm long, 30 mm above the board
MEM_INNER = 0.027  # the inner module's face toward the processor, from the processor's centre (along -x)
MEM_PITCH = 0.008
MEM_X = (CPU_X - MEM_INNER - MEM_SIZE[0] / 2, CPU_X - MEM_INNER - MEM_PITCH - MEM_SIZE[0] / 2)  # 0.271, 0.263
MEM_TOP = BOARD_TOP + MEM_SIZE[2]  # 34 mm

# The VRM heatsink on the processor's left (+y), along x.
VRM_SIZE = (0.070, 0.014, 0.026)
VRM_INNER = 0.046  # its face toward the processor, from the processor's centre (along +y)
VRM_X, VRM_Y = CPU_X, CPU_Y + VRM_INNER + VRM_SIZE[1] / 2  # (0.300, 0.053)
VRM_TOP = BOARD_TOP + VRM_SIZE[2]  # 30 mm

# The graphics card standing in the top slot on the processor's right (-y), its long side along x.
GPU_SIZE = (0.120, 0.020, 0.050)
GPU_INNER = 0.046  # its face toward the processor, from the processor's centre (along -y)
GPU_X, GPU_Y = 0.310, CPU_Y - GPU_INNER - GPU_SIZE[1] / 2  # (0.310, -0.056)
GPU_TOP = BOARD_TOP + GPU_SIZE[2]  # 54 mm

# The rear I/O shield along the board's far edge.
IO_SIZE = (0.010, 0.090, 0.040)
IO_X, IO_Y = 0.378, 0.000  # its face toward the processor at x 0.373 (73 mm from the processor's centre)
IO_TOP = BOARD_TOP + IO_SIZE[2]  # 44 mm

# -- the staging pad and the cooler ------------------------------------------------------------------------------
PAD_SIZE = (0.080, 0.080, 0.010)  # pink anti-static foam, fixed
PAD_X, PAD_Y = 0.180, 0.235
PAD_TOP = PAD_SIZE[2]

# The tower cooler: one loose rigid body (mechkit compound). Base frame: origin at the centre of the base's bottom
# face; local +x is the airflow (from the fan through the fins), local +z up. The fin stack is offset 'downstream'
# of the base, so along the airflow it reaches 37 mm from the base's centre one way and 18 mm the other (20 mm and
# 1 mm past the base's edges), and 37 mm to either side across it.
BASE_SIZE = (0.034, 0.034, 0.006)  # copper base
BASE_HALF = BASE_SIZE[0] / 2
BASE_CENTER = (0.0, 0.0, BASE_SIZE[2] / 2)
RISER_SIZE = (0.020, 0.020, 0.014)  # copper block standing in for the heat pipes, base top to the fins' foot
RISER_CENTER = (0.0, 0.0, BASE_SIZE[2] + RISER_SIZE[2] / 2)
FOOT_UP = BASE_SIZE[2] + RISER_SIZE[2]  # 20 mm: the fan's and fins' lowest edge above the base's bottom face
STACK_H = 0.050
STACK_W = 0.074  # across the airflow
UPSTREAM = 0.018  # the fan's intake face, behind the base's centre (along -airflow)
DOWNSTREAM = 0.037  # the fins' outlet face, ahead of the base's centre (along +airflow)
FAN_T = 0.012
FAN_SIZE = (FAN_T, STACK_W, STACK_H)
FAN_CENTER = (-UPSTREAM + FAN_T / 2, 0.0, FOOT_UP + STACK_H / 2)  # (-0.012, 0, 0.045)
FINS_SIZE = (UPSTREAM + DOWNSTREAM - FAN_T, STACK_W, STACK_H)  # 43 mm deep
FINS_CENTER = (-UPSTREAM + FAN_T + FINS_SIZE[0] / 2, 0.0, FOOT_UP + STACK_H / 2)  # (0.0155, 0, 0.045)
COOLER_H = FOOT_UP + STACK_H  # 70 mm
COOLER_MASS = 0.25
STACK_MID = (DOWNSTREAM - UPSTREAM) / 2  # 9.5 mm: the middle of fan plus fins along the airflow, from the base centre
TOP_LOCAL = (STACK_MID, 0.0, COOLER_H)  # the centre of the fin stack's top, followed for 'at rest'
# Mass is spread over the colliders by volume (PhysX does this when only a mass is given): mechkit puts the centre
# of mass 9.0 mm downstream of the base's centre and 42.6 mm up, 8.0 mm inside the base's 17 mm half-width, so the
# cooler stands on its base (it would tip past about 10.6 degrees) on the pad and on the processor.
START = (PAD_X, PAD_Y, TOP + PAD_TOP)  # the cooler's base frame at the start (world)
START_YAW = math.pi / 2  # airflow along +y: a quarter turn from the one way it fits

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------------
SEAT_Z_TOL = 0.002  # 'seated': the base's bottom-face centre within 2 mm of the processor's lid
SEAT_TILT_DEG = 2.0  # ... the base tilted 2 degrees or less
AIRFLOW_TOL_DEG = 15.0  # ... the airflow within 15 degrees of +x (toward the rear exhaust); also needed for 0.5
# ... and the base's four bottom corners all inside the frame's opening (44 x 44 mm)
NEAR = 0.020  # 0.5: the base's centre within 20 mm of the processor's centre, horizontally (and the right way round)
UPRIGHT_DEG = 30.0  # 0.25 and 0.5 need the cooler tilted 30 degrees or less
STILL_WINDOW_S = 0.5  # 'at rest': the base's centre and the fin stack's top each moved 2 mm or less in 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the cooler moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone more than this from the cooler's box: it let go

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
PCB_GREEN = (0.10, 0.36, 0.20)
LID_GREY = (0.74, 0.75, 0.77)
FRAME_BLACK = (0.05, 0.05, 0.05)
MEM_BLACK = (0.09, 0.09, 0.10)
VRM_GREY = (0.32, 0.33, 0.35)
GPU_BLACK = (0.14, 0.14, 0.15)
IO_GREY = (0.58, 0.59, 0.61)
FOAM_PINK = (0.90, 0.56, 0.66)
COPPER = (0.78, 0.47, 0.25)
ALUMINIUM = (0.82, 0.83, 0.85)
FAN_BLACK = (0.08, 0.08, 0.08)
