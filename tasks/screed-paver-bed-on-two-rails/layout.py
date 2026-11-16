"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights named h_* or *_H are above the table top.

The work: a paver bedding course 400 mm (along y) by 200 mm (along x), the bed of two pavers, inside an edging of
three steel flats (far, left, right). Two steel rails lie along x at y +-112.5 mm, their tops 25 mm up: the thickness
the bedding course must have. The bedding sand starts 36 mm deep with a mound and a hollow. A loose aluminium screed
board, standing on edge, rides the rails and cuts the sand to their height as it is pulled toward the worker. It
starts on two rest blocks beside the bed (it is fetched from there) and ends on them again.
"""

import math

TOP = 0.7475  # station table top (world z)
G = 9.81
MM = 0.001

# -- the bed: a grid of 25 mm sand cells --------------------------------------------------------------------------
CELL = 0.025
NX, NY = 8, 16  # cells along x (near edge to far edge) and along y
BED_X0 = 0.200  # the open near edge, toward the robot
BED_X1 = BED_X0 + NX * CELL  # 0.400: where the scored cells end
BED_Y0 = -NY * CELL / 2  # -0.200
BED_Y1 = NY * CELL / 2  # +0.200
SLIVER = 0.015  # 15 mm between the last cell and each wall (raked flush at 25 mm, never reached: not scored)
INSIDE_X1 = BED_X1 + SLIVER  # 0.415: the far wall's inner face
INSIDE_Y = BED_Y1 + SLIVER  # +-0.215: the side walls' inner faces

# -- the edging (fixed steel flats, 25 mm thick, 60 mm tall) -----------------------------------------------------------
WALL_T = 0.025
WALL_H = 0.060
WALL_X1 = INSIDE_X1 + WALL_T  # 0.440: the far edge of everything
GREY = (0.55, 0.57, 0.60)

# -- the rails (fixed: 25 x 25 mm steel square tube, bedded in the sand) -------------------------------------------------
RAIL_W = 0.025  # across (along y): exactly one cell column
RAIL_H = 0.025  # tall: the bedding thickness
RAIL_Y = 0.1125  # centre lines y +-112.5 mm: they fill the cell columns 100..125 mm out
RAIL_X0, RAIL_X1 = BED_X0, INSIDE_X1  # 0.200 .. 0.415
ORANGE = (0.90, 0.45, 0.10)
RAIL_COLS = (3, 12)  # the cell columns the rails fill (y -125..-100 and +100..+125)

# -- the drawn sand (fixed, 20 mm: a stand-in for the sand under the sole; the height model above it is not drawn) -------
SAND_DRAW_H = 0.020
SAND_COLOR = (0.86, 0.76, 0.52)

# -- the sand (the scorer's height model) ------------------------------------------------------------------------------
BED_TARGET = 0.025  # the rail tops: the depth every cell must reach
START_H = 0.036  # the raked start depth
MOUND_H = 0.045  # a mound, 9 mm over the start depth
HOLLOW_H = 0.019  # a hollow, 6 mm under the rails' tops
MOUND_I, MOUND_J = range(4, 7), range(4, 7)  # 3 x 3 cells: x 300..375 mm, y -100..-25 mm
HOLLOW_I, HOLLOW_J = range(1, 4), range(8, 11)  # 3 x 3 cells: x 225..300 mm, y 0..+75 mm
UNDER_BOARD_I = NX - 1  # the column under the board as it starts: already cut to 25 mm
SAND_DENSITY = 1500.0  # kg/m3 (play sand, est): about 3.6 kg in the bed

# -- the screed board (loose: 25 x 50 mm aluminium tube, 1.5 mm wall, 400 mm long, on edge) -------------------------------
BD_X = 0.025  # across: the sole's width (along x)
BD_Z = 0.050  # tall
BD_L = 0.400  # along y
BD_MASS = 0.24  # kg: 25x50x1.5 wall tube, 225 mm2 x 400 mm x 2700 kg/m3, modelled as a solid box of that mass
BD_COLOR = (0.78, 0.79, 0.82)
BD_START_X = 0.3875  # centre when it starts on the rails: sole 375 .. 400 mm, over the last cell column
BD_END_X = 0.2125  # centre at the end of the pull: sole 200 .. 225 mm, over the first cell column
PULL = BD_START_X - BD_END_X  # 0.175 m

# -- the rest blocks (fixed steel blocks beside the bed: where the board is kept) ------------------------------------------
STOP_X = 0.150  # centre x: 50 mm short of the bed's near edge
STOP_Y = 0.140  # centres y +-140 mm
STOP_SIZE = (0.040, 0.030, 0.055)  # along x, along y, tall
STOP_H = STOP_SIZE[2]
STOP_COLOR = (0.30, 0.30, 0.33)
BD_REST_Z = STOP_H  # the board's sole when it rests on the blocks
BD_REST_X = STOP_X

# -- scoring thresholds (scoring.py) --------------------------------------------------------------------------------------
TALL_MAX = 0.032  # reported only (details "tall"): no cell deeper than this
CUT_CELLS = 84  # tier 0.25: at least this many of the 112 sand cells (three quarters) within TOL_LOOSE of 25 mm
TOL_LOOSE, TOL_MED, TOL_TIGHT = 0.006, 0.004, 0.003  # every cell within this of 25 mm
BRIDGE_TOL = 0.003  # 'bridged': the sole within this of both rails' tops
BRIDGE_PULL = 0.150  # a pull at least this long, in one go
RIDE_TOL = 0.003  # the board fills a hollow only while its sole is within this of the rails' tops
STOWED_Z_TOL = 0.003  # 'stowed': the sole within this of the blocks' tops at both blocks
STOWED_XY_TOL = 0.010  # centre within this of the blocks' centre line
STOWED_YAW_DEG = 3.0
STOWED_TILT_DEG = 3.0
TIP_DEG = 30.0  # tilted more than this (on its side or flat) the board cuts nothing; touching finished sand presses it
OFF_TABLE_DROP = 0.020  # latched: the board's lowest point this far under the table top
TOUCH_TOL = 0.002  # a pad tip this close to a finished cell's sand, or in it, touched it
CONTACT_TOL = 0.0005  # the sole is on a cell's sand when it is within this of it
LET_GO = 0.030  # hands off: every finger zone at least 30 mm from the board, the bed and the rails
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001  # the board moves under 1 mm and turns under 0.3 deg over the last second
STILL_TURN_DEG = 0.3
BED_TOP = 0.045  # the box standing for the bed in the hands-off distance: 45 mm tall, 2 mm over the highest sand
CONTROL_HZ = 20.0
RESET_JUMP = 0.08  # the board back on its starting mark having moved this much in one control step: a new episode
START_TOL = 0.005
EPS = 1e-9


def cell_x(i):
    return BED_X0 + (i + 0.5) * CELL


def cell_y(j):
    return BED_Y0 + (j + 0.5) * CELL


def start_heights():
    """The start depth of every cell (nested lists [i][j], m): 36 mm, a 3 x 3 mound to 45 mm, a 3 x 3 hollow to
    19 mm, the column under the board and the two rail columns at 25 mm."""
    h = [[START_H] * NY for _ in range(NX)]
    for i in MOUND_I:
        for j in MOUND_J:
            h[i][j] = MOUND_H
    for i in HOLLOW_I:
        for j in HOLLOW_J:
            h[i][j] = HOLLOW_H
    for j in range(NY):
        h[UNDER_BOARD_I][j] = BED_TARGET
    for i in range(NX):
        for j in RAIL_COLS:
            h[i][j] = BED_TARGET
    return h


BOARD_START = (BD_REST_X, 0.0, TOP + BD_REST_Z + BD_Z / 2)  # the board's centre at the start
