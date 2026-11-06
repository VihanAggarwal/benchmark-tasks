"""Every dimension and threshold of the task, in one place (metres, kilograms, degrees where named, station frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The board lies across the table, long axis along y: its nose is the left end (+y), its tail the right end (-y), and
its heel edge (x 0.180) faces the robot. A binding's frame: origin at the centre of its baseplate's underside (the
disc's centre, "the binding's centre"), u (its x axis) from heel to toe, v across, z up. Its angle is the heading of
u seen from above, in degrees from +x (straight across the board, away from the robot), positive toward the nose.
"""

TOP = 0.7475  # station table top

# -- the board: three fixed strips 10 mm thick with one flush top ---------------------------------------------------
BOARD_T = 0.010
BOARD_TOP = TOP + BOARD_T  # 0.7575
BOARD_X0, BOARD_X1 = 0.180, 0.420  # heel edge (toward the robot), toe edge
BOARD_Y0, BOARD_Y1 = -0.550, 0.550  # tail end, nose end
BOARD_LEN = BOARD_Y1 - BOARD_Y0  # 1.100
LINE_X = 0.300  # the black centre line along the board
LINE_W = 0.004
STRIP_W = 0.118  # each white strip, heel side and toe side of the line
HEEL_STRIP_X = BOARD_X0 + STRIP_W / 2  # 0.239
TOE_STRIP_X = BOARD_X1 - STRIP_W / 2  # 0.361
BOARD_STRIPS = (  # (name, centre x, width along x, colour key)
    ("board_heel", HEEL_STRIP_X, STRIP_W, "white"),
    ("board_line", LINE_X, LINE_W, "black"),
    ("board_toe", TOE_STRIP_X, STRIP_W, "white"),
)

# -- the red reference cross: two thin fixed strips, the one along y stacked on the one along x --------------------
CROSS_X, CROSS_Y = LINE_X, 0.0
CROSS_LEN, CROSS_W, CROSS_T = 0.030, 0.004, 0.0005

# -- the insert grids: 16 fixed grey dots, 8 mm across, 0.5 mm proud (vinyl dot stickers) ------------------------------
DOT_R, DOT_H = 0.004, 0.0005
DOT_XS = (0.280, 0.320)
DOT_YS_NOSE = (0.14, 0.18, 0.22, 0.26)  # nose grid, centred at y +0.20
DOT_YS_TAIL = (-0.14, -0.18, -0.22, -0.26)  # tail grid, centred at y -0.20
# 'Over an insert grid': the disc centre between a grid's first and last rows along the board, |y| 0.140 to 0.260
# (either grid; centred on the line, it is then between the grid's two columns as well).
GRID_Y = (min(abs(y) for y in DOT_YS_NOSE), max(abs(y) for y in DOT_YS_NOSE))  # (0.14, 0.26)

# -- the bindings: two identical loose rigid compounds (mechkit assemblies) -----------------------------------------
BINDING_MASS = 0.25
# (part, kind, centre in the binding frame, size: box edges along u, v, z / cylinder (radius, height), colour key)
BASEPLATE = ((0.0, 0.0, 0.0075), (0.200, 0.120, 0.015))
DISC_CENTER, DISC_R, DISC_H = (0.0, 0.0, 0.0165), 0.045, 0.003
TICK = ((0.032, 0.0, 0.0185), (0.020, 0.004, 0.001))  # white tick on the disc, pointing to the toe
HIGHBACK = ((-0.095, 0.0, 0.055), (0.010, 0.120, 0.080))  # u -0.100..-0.090, z 0.015..0.095
# The ankle-strap bar's centre along u: over the disc's centre. The binding's centre of mass is at u -0.0147, z 0.0219
# (under the bar, near its heel face). With the bar 20 mm further back, over the centre of mass, the right gripper's
# housing (offset 27 mm to one side of the closing axis) would pass 9 mm from the tail binding's highback; here 27 mm.
STRAP_U = 0.0
STRAP = ((STRAP_U, 0.0, 0.0365), (0.030, 0.130, 0.037))  # z 0.018..0.055, resting on the disc
STRAP_TOP = 0.055
# Every part as a box in the binding frame (the disc as its bounding box): what 'let go' measures to.
PART_BOXES = (
    ("baseplate", BASEPLATE[0], BASEPLATE[1]),
    ("disc", DISC_CENTER, (2 * DISC_R, 2 * DISC_R, DISC_H)),
    ("tick", TICK[0], TICK[1]),
    ("highback", HIGHBACK[0], HIGHBACK[1]),
    ("strap", STRAP[0], STRAP[1]),
)
# The offline checker's stand-in (task.Binding.offline_box): the strap's plan footprint from the underside to the
# strap's top, the column a gripper closes on. The simulator never uses it.
COLUMN_CENTER = (STRAP_U, 0.0, STRAP_TOP / 2)  # (0, 0, 0.0275)
COLUMN_SIZE = (0.030, 0.130, STRAP_TOP)

# Start: each binding parked flat on its own end of the board, disc on the centre line, toe toward that end and
# 5 mm from it. (name, origin x, origin y, angle in degrees)
NOSE_START = (LINE_X, 0.445, 90.0)
TAIL_START = (LINE_X, -0.445, -90.0)

# -- the stance --------------------------------------------------------------------------------------------------
STANCE_W = 0.400  # disc centre to disc centre along the board
FRONT_TARGET_DEG = 15.0  # the binding nearer the nose: toe turned toward the nose
BACK_TARGET_DEG = -15.0  # the other: toe turned toward the tail

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------------
FLAT_Z_TOL = 0.003  # the binding's underside centre within 3 mm of the board's top (the 0.5 mm dots allowed)
FLAT_TILT_DEG = 5.0  # its z axis within 5 deg of vertical (upside down is 180)
CENTRE_TOL = 0.010  # disc centre within 10 mm of the centre line (x 0.290..0.310)
ANGLE_TOL_DEG = 3.0  # one 3-degree step of the disc's angle scale
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
STILL_WINDOW_S = 0.5  # stillness is judged over the last 0.5 s (10 control steps), by path, never by velocity
STILL_PATH = 0.002  # the centre's path over the window
STILL_TURN_DEG = 1.0  # the turn (summed heading change) over the window
LET_GO = 0.030  # each gripper's finger zone (below) at least 30 mm from every part of the binding
W_TIGHT = (0.390, 0.410)  # 1.0: the right insert position
C_TIGHT = 0.010
W_LOOSE = (0.380, 0.420)  # 0.75: within 2 cm of the chart width and of the cross
C_LOOSE = 0.020
# The grippers' finger zone: rigstate gives no gripper heading, so the fingers are taken to be anywhere round the
# gripper's axis. The zone is every point within a radius of the 38 mm segment from the point midway between the
# finger tips back up the gripper (toward the TCP); the radius reaches a finger pad's outer corner: half the current
# finger gap plus the pad's 8 mm thickness, and half its 27 mm width across.
# 56.6 mm fully open (94 mm), 45.1 mm at the demo's 70 mm, 15.7 mm closed.
FULL_GAP = 0.0939  # the pads' gap fully open (rigstate opening 1.0)
FINGER_THICK = 0.008
FINGER_HALF_WIDTH = 0.0135
FINGER_LEN = 0.038
ZONE_SAMPLES = 39  # points along the 38 mm segment, 1 mm apart
EPS = 1e-9  # m: comparisons at a boundary count as inside (floating-point slack only)
EPS_DEG = 1e-6
# Fallback reset detection (rigstate's and mechkit's reset hooks come first): both bindings put back on their start
# poses (within 5 mm and 2 deg) in one control step, at least one of them jumping 50 mm or turning 20 deg to get
# there. A carried binding moves at most about 5 mm per step.
RESET_START_TOL = 0.005
RESET_START_TURN_DEG = 2.0
RESET_JUMP = 0.050
RESET_TURN_DEG = 20.0

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
WHITE = (0.95, 0.95, 0.93)
BLACK = (0.05, 0.05, 0.05)
RED = (0.85, 0.12, 0.12)
DOT_GREY = (0.55, 0.55, 0.58)
DISC_GREY = (0.62, 0.62, 0.64)
STRAP_GREY = (0.45, 0.45, 0.48)
TICK_WHITE = (0.97, 0.97, 0.97)
COLORS = {"white": WHITE, "black": BLACK}
