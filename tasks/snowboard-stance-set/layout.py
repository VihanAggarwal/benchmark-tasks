"""Every dimension and threshold of the task, in one place (metres, kilograms, degrees where named, station frame).

Real size throughout: a child's snowboard top with M6 inserts on the standard 4 x 4 pattern (40 mm grid), two
child's bindings with 4 x 4 discs, eight M6 binding screws with washers.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The board lies across the table, long axis along y: its nose is the left end (+y), its tail the right end (-y), and
its heel edge (x 0.180) faces the robot.

A binding's frame is its DISC's frame: origin at the disc's centre on the underside, x and y square to the disc's
slot pattern, z up. The baseplate (heel to toe along u) is turned PRESET degrees in that frame: +15 on the nose
binding, -15 on the tail binding, as a shop presets a rental binding's disc for a duck stance. The binding's angle
is the heading of u seen from above, in degrees from +x (straight across the board), positive toward the nose; it
equals the disc frame's heading plus PRESET. When the disc frame is square to the board (heading 0), its four
slots sit over a 40 mm square of inserts and the binding's angle is PRESET.

A screw's frame: origin on its axis at the washer's underside (the face that bears on the disc), z up the axis.
"""

import math

TOP = 0.7475  # station table top

# -- the board: a fixed plank 10 mm thick with 16 insert bores ----------------------------------------------------
BOARD_T = 0.010
BOARD_TOP = TOP + BOARD_T  # 0.7575
BOARD_X0, BOARD_X1 = 0.180, 0.420  # heel edge (toward the robot), toe edge
BOARD_Y0, BOARD_Y1 = -0.550, 0.550  # tail end, nose end
BOARD_W = BOARD_X1 - BOARD_X0  # 0.240
BOARD_LEN = BOARD_Y1 - BOARD_Y0  # 1.100
BOARD_CX = (BOARD_X0 + BOARD_X1) / 2  # 0.300: the board assembly's frame origin (on the table, at y 0)
LINE_X = 0.300  # the black centre line along the board
LINE_W = 0.004
STRIP_W = (BOARD_W - LINE_W) / 2  # 0.118: each white strip, heel side and toe side of the line
FLOOR_LAYER = 0.001  # the board's bottom millimetre, dark: it is each bore's floor (the insert's end)
INSERT_HOLE = 0.008  # each insert's bore, square 8 mm (the M6 thread inside it is not modelled)
INSERT_DEPTH = BOARD_T - FLOOR_LAYER  # 9 mm
INSERT_XS = (0.280, 0.320)  # the two rows across the board, 40 mm apart, either side of the centre line
INSERT_YS_NOSE = (0.140, 0.180, 0.220, 0.260)  # four rows along the board, 40 mm apart: nose half
INSERT_YS_TAIL = tuple(-y for y in INSERT_YS_NOSE)
INSERTS = tuple((f"{half}{k + 1}{'h' if x < LINE_X else 't'}", x, y)
                for half, ys in (("n", INSERT_YS_NOSE), ("t", INSERT_YS_TAIL)) for k, y in enumerate(ys)
                for x in INSERT_XS)  # 16: name (n/t, row 1..4 from the middle, h heel / t toe side), x, y
INSERT_PITCH = 0.040
# The squares a 4 x 4 disc can be screwed to: two adjacent rows of one half. Disc centre |y| 0.160, 0.200, 0.240.
SQUARE_YS = {"n": (0.160, 0.200, 0.240), "t": (-0.160, -0.200, -0.240)}
MIDDLE_SQUARE_Y = 0.200  # the 40 cm stance centred on the cross: discs at y +0.200 and -0.200

# -- the red reference cross: two thin fixed strips, the one along y stacked on the one along x --------------------
CROSS_X, CROSS_Y = LINE_X, 0.0
CROSS_LEN, CROSS_W, CROSS_T = 0.030, 0.004, 0.0005

# -- the bindings: two loose rigid parts, discs preset to +15 (nose) and -15 (tail) ----------------------------------
BINDING_MASS = 0.35  # a small child's binding; 0.40 sags past the slots' play when hung by its highback
PRESET_DEG = {"binding_nose": 15.0, "binding_tail": -15.0}
FLOOR_T = 0.008  # baseplate floor and disc, flush: the disc's top is 8 mm above the binding's underside
BASE_LEN, BASE_W = 0.230, 0.130  # baseplate along u (heel to toe) and v
DISC_OPENING = 0.092  # the baseplate's opening the disc sits in (square, baseplate frame)
DISC_CORE = 0.064  # the disc: a 64 mm square core with the slots, plus four tabs out to 44 mm (an 88 mm disc)
DISC_TAB = (0.012, 0.050)  # each tab: 12 mm deep, 50 mm long
DISC_TAB_AT = 0.038  # tab centres, from the disc's centre
SLOT_CENTRES = ((0.020, 0.020), (-0.020, 0.020), (-0.020, -0.020), (0.020, -0.020))  # disc frame: a 40 mm square
SLOT_LEN, SLOT_W = 0.024, 0.010  # each slot: 24 mm along the disc's x (heel-to-toe centring), 10 mm along y
SIDEWALL_U = (-0.105, 0.075)  # along u, from the highback to 40 mm short of the toe end
SIDEWALL_T, SIDEWALL_H = 0.006, 0.040
SIDEWALL_V = BASE_W / 2 - SIDEWALL_T / 2  # 0.062: each sidewall's centre line
HIGHBACK_U = (-0.115, -0.105)  # 10 mm thick at the heel end
HIGHBACK_H = 0.100  # above the floor: its top is 108 mm above the underside
HIGHBACK_TOP = FLOOR_T + HIGHBACK_H  # 0.108
HIGHBACK_W = BASE_W  # 0.130
# Start: each binding parked flat on its own end of the board, disc on the centre line, toe toward that end and 5 mm
# from it. (disc x, disc y, binding angle in degrees)
NOSE_START = (LINE_X, 0.430, 90.0)
TAIL_START = (LINE_X, -0.430, -90.0)

# -- the screws: eight loose M6 binding screws with washers --------------------------------------------------------
SCREW_MASS = 0.006  # steel, with its washer
WASHER_R, WASHER_T = 0.009, 0.0015  # 18 mm washer
HEAD_R, HEAD_H = 0.006, 0.005  # 12 mm pan head
STUB_R = 0.0025  # the M6 thread's 5 mm core (the thread itself is not modelled)
STUB_L = 0.0145  # below the washer: an M6 x 16 under its head, less the 1.5 mm washer
# Two fixed screw holders (drilled blocks) on the table beyond the board's toe edge, four screws standing head up
# in each: the nose-side holder for the left arm, the tail-side holder for the right arm.
HOLDER_X = 0.455
HOLDER_YS = (0.215, 0.245, 0.275, 0.305)  # socket rows, nose side (the tail side mirrored)
HOLDER_SIZE = (0.030, 0.120, 0.024)
HOLDER_TOP = 0.024  # above the table
HOLDER_SOCKET = 0.006  # 6 mm sockets: the 5 mm stubs stand within 0.5 mm of upright and of the socket's centre
HOLDER_BASE_T = 0.004  # solid below the 20 mm deep sockets

# -- the rules: a screw dropped through a slot into an insert and turned while seated ------------------------------
ENGAGE_GAP = 0.0010  # the thread starts (the turn count starts) with the washer within 1 mm of the disc's top ...
DISENGAGE_GAP = 0.0030  # ... and lets go (the count goes back to 0) once the washer is 3 mm up, or the stub leaves
STUB_DEPTH_MIN = 0.005  # the stub's tip at least 5 mm down the insert
SCREW_TILT_DEG = 5.0  # upright within 5 deg (engaging: 10 deg)
ENGAGE_TILT_DEG = 10.0
FIT_SLACK = 0.0005  # contact offset and rounding: a stub counts as in a bore or slot up to 0.5 mm past its free play
DRIVE_TURN_DEG = 360.0  # one full turn clockwise (seen from above) while seated: driven home, then welded
# -- the end-state rules (scoring.py) ------------------------------------------------------------------------------
FLAT_Z_TOL = 0.0015  # a binding's underside centre within 1.5 mm of the board's top ...
FLAT_TILT_DEG = 3.0  # ... and its z axis within 3 deg of vertical
ANGLE_TOL_DEG = 3.0  # its angle within 3 deg of its target (one step of a disc's angle scale)
FRONT_TARGET_DEG = 15.0  # the binding nearer the nose: toe turned toward the nose
BACK_TARGET_DEG = -15.0  # the other: toe turned toward the tail
SEAT_GAP = 0.0010  # a seated screw: its washer within 1 mm of the disc's top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
STILL_WINDOW_S = 1.0  # every part still and both grippers released on every step of the last 1 s (20 steps)
STILL_PATH = 0.002  # each part's origin travels 2 mm or less over the window ...
STILL_TURN_DEG = 1.0  # ... and turns 1 deg or less in total
LET_GO = 0.030  # each gripper's finger zone at least 30 mm from every binding and every screw
LOST_Z = TOP - 0.020  # a binding or screw below this has fallen off the table: the trial scores 0 from then on
EPS = 1e-9  # m: comparisons at a boundary count as inside (floating-point slack only)
EPS_DEG = 1e-6
# Fallback reset detection (rigstate's and mechkit's reset hooks come first): both bindings put back on their start
# poses (within 5 mm and 2 deg) in one control step, at least one of them jumping 50 mm or turning 20 deg.
RESET_START_TOL = 0.005
RESET_START_TURN_DEG = 2.0
RESET_JUMP = 0.050
RESET_TURN_DEG = 20.0

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
WHITE = (0.95, 0.95, 0.93)
BLACK = (0.05, 0.05, 0.05)
RED = (0.85, 0.12, 0.12)
INSERT_GREY = (0.30, 0.30, 0.33)
DISC_GREY = (0.62, 0.62, 0.64)
STEEL = (0.78, 0.78, 0.80)
HOLDER_BLUE = (0.20, 0.28, 0.45)


def preset_rad(name: str) -> float:
    return math.radians(PRESET_DEG[name])


def middle_inserts(half: str):
    """The four inserts of a half's middle square (the 40 cm stance): (x, y) pairs."""
    ys = (0.180, 0.220) if half == "n" else (-0.180, -0.220)
    return tuple((x, y) for y in ys for x in INSERT_XS)
