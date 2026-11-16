"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A kraft mailing tube (51 mm inside, 54 mm outside, 457 mm = 18 in long, open at both ends) lies along y on a packing
board. A 38 mm paper roll lies on the table in line with the tube's left end. Two black plastic end caps lie on their
sides in two cap rests, one in front of each arm, plug toward the tube. Goal: the roll inside the tube and a cap
pressed on each end, hands off.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
IN = 0.0254
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the packing board (fixed) and the tube (loose) -------------------------------------------------------------------
BOARD_X = 0.30  # the tube's, the board's and the roll's centre line
BOARD_LX, BOARD_LY, BOARD_T = 0.100, 0.400, 0.005  # a 5 mm packing board, shorter than the tube: caps overhang it
BOARD_COLOR = (0.62, 0.50, 0.36)
BOARD_FRICTION = (0.5, 0.45)

TUBE_L = 18 * IN  # 457.2 mm
TUBE_ID, TUBE_OD = 0.051, 0.054  # 51 mm bore, 54 mm outside (1.5 mm kraft board)
TUBE_WALL = (TUBE_OD - TUBE_ID) / 2
TUBE_R_IN = TUBE_ID / 2  # 25.5 mm: the bore's apothem
TUBE_R_OUT = TUBE_OD / 2  # 27.0 mm
TUBE_STAVES = 16  # the wall is a 16-sided ring of flat staves (a round tube is not a native shape)
TUBE_STAVE_W = 2 * TUBE_R_OUT * math.tan(math.pi / TUBE_STAVES) + 0.0015  # a little wider than a side so corners meet
TUBE_MASS = 0.12  # kraft board, est
TUBE_COLOR = (0.70, 0.55, 0.36)
TUBE_FRICTION = (0.40, 0.35)  # kraft paper on the board (est)
TUBE_Z = BOARD_T + TUBE_R_OUT  # the tube's axis over the table: 32 mm (it rests on its flat staves on the board)
TUBE_END = TUBE_L / 2  # an end's distance from the tube's centre along its axis

# -- the poster roll (loose) --------------------------------------------------------------------------------------------
ROLL_R = 0.019  # 38 mm
ROLL_L = 0.380  # 15 in poster, short enough that both end caps' plugs clear its ends
ROLL_MASS = 0.060  # est
ROLL_COLOR = (0.93, 0.92, 0.88)
ROLL_FRICTION = (0.40, 0.35)
ROLL_LEAD_Y = 0.245  # the roll's leading (right) end as found: 16.5 mm outside the tube's left mouth
ROLL_Y0 = ROLL_LEAD_Y + ROLL_L / 2  # its centre as found
ROLL_Z0 = ROLL_R  # lying on the table

# -- the end caps (loose) -----------------------------------------------------------------------------------------------
# Base frame at the centre of the flange's inner face; local +y points out of the tube (flange side); the plug is at
# local y < 0: a body 49 mm across and 15 mm long ending in a 45 mm lead-in nose 3 mm long.
CAP_FLANGE_R, CAP_FLANGE_T = 0.030, 0.005  # 60 mm flange, 5 mm thick
CAP_PLUG_R, CAP_PLUG_L = 0.0245, 0.015  # 0.5 mm radial play in the 51 mm bore at the flats (est: a slip fit)
CAP_NOSE_R, CAP_NOSE_L = 0.0225, 0.003
CAP_DEPTH = CAP_PLUG_L + CAP_NOSE_L  # 18 mm of plug
CAP_MASS = 0.012
CAP_COLOR = (0.10, 0.10, 0.12)
CAP_FRICTION = (0.5, 0.45)
CAP_Z0 = CAP_FLANGE_R  # lying on its flange rim on the table
CAP_X0 = 0.17  # where each cap lies (in front of its own arm)
CAP_Y0 = {"a": 0.345, "b": -0.345}  # flange inner face centre as found (cap a on the left, cap b on the right)
CAP_YAW = {"a": 0.0, "b": math.pi}  # local +y toward world +y for a, -y for b (flange away from the tube)

# the cap rest: a low platform under the plug, so the cap lies level, and a rail each side
REST_PLAT_T = CAP_Z0 - CAP_PLUG_R - 0.0005  # 5.0 mm
REST_PLAT_X, REST_PLAT_Y = 0.080, 0.014
REST_RAIL_H, REST_RAIL_W, REST_RAIL_L = 0.012, 0.008, 0.030
REST_RAIL_X = CAP_FLANGE_R + 0.004 + REST_RAIL_W / 2  # 4 mm clear of the flange
REST_COLOR = (0.42, 0.42, 0.46)

# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------------
ROLL_END_IN = 0.015  # each end of the roll at least this far inside the tube's end
ROLL_RADIAL = 0.008  # and within this of the tube's axis (the bore leaves 6.5 mm)
CAP_GAP = 0.0015  # a cap's flange inner face within this of the tube's end (outside it)
CAP_PEN = 0.002  # ... or this far into it
CAP_TILT_DEG = 5.0  # cap axis within this of the tube's, pointing out of the tube
CAP_RADIAL = 0.0025  # flange centre within this of the tube's axis
CAP_NOSE_RADIAL = 0.0030  # the plug's tip within this of the tube's axis: the plug is in the bore
FALL_Z = -0.030  # a part's centre lower than this (off the table) is lost
TABLE_Y, TABLE_X0, TABLE_X1 = 0.65, -0.19, 0.91
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001
STILL_TURN_DEG = 2.0  # GPU: seated 12 g caps jitter 1.0-1.4 deg in the bore (solver contact slop, <0.2 mm at the rim)
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9

PARTS = ("tube", "roll", "cap_a", "cap_b")
CAP_NAMES = {"a": "cap_a", "b": "cap_b"}


def tube_origin():
    return (BOARD_X, 0.0, TUBE_Z)


def roll_origin():
    return (BOARD_X, ROLL_Y0, ROLL_Z0)


def cap_origin(side: str):
    return (CAP_X0, CAP_Y0[side], CAP_Z0)
